import logging
from typing import Dict

import torch
import torch.nn as nn

from neuralhydrology.datautils.utils import get_frequency_factor, sort_frequencies
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.modelzoo.inputlayer import InputLayer
from neuralhydrology.utils.config import Config

LOGGER = logging.getLogger(__name__)


class MTSSequentialForecastLSTM(BaseModel):
    """A forecasting model that combines MTSLSTM with a sequential forecast LSTM. 
    
    This model processes inputs at multiple timescales, passing states from lower to 
    higher frequency LSTMs. At the highest frequency, it uses a sequential LSTM 
    that processes a hindcast period followed by a forecast period.
    
    Parameters
    ----------
    cfg : Config
        The run configuration.
    """
    module_parts = ['hindcast_embedding_net', 'forecast_embedding_net', 'lstms', 'transfer_fcs', 'heads']

    def __init__(self, cfg: Config):
        super(MTSSequentialForecastLSTM, self).__init__(cfg=cfg)

        if cfg.forecast_overlap:
            raise ValueError('Forecast overlap cannot be set for a sequential forecasting model.')

        # Part from MTSLSTM
        self._slice_timestep = {}
        self._frequency_factors = []
        self._seq_lengths = cfg.seq_length
        self._is_shared_mtslstm = self.cfg.shared_mtslstm
        if self._is_shared_mtslstm: 
            raise ValueError(f'sMTS-LSTM is not supported')

        self._transfer_mtslstm_states = self.cfg.transfer_mtslstm_states
        transfer_modes = [None, "None", "identity", "linear"]
        if self._transfer_mtslstm_states["h"] not in transfer_modes \
                or self._transfer_mtslstm_states["c"] not in transfer_modes:
            raise ValueError(f"MTS-LSTM supports state transfer modes {transfer_modes}")

        if len(cfg.use_frequencies) < 2:
            raise ValueError("MTS-Forecast-LSTM expects at least two frequencies.")
        self._frequencies = sort_frequencies(cfg.use_frequencies)
        self._highest_freq = self._frequencies[-1]

        # Embedding networks
        self.hindcast_embedding_net = nn.ModuleDict()
        self.forecast_embedding_net = nn.ModuleDict()

        for freq in self._frequencies:
            # Create a config for each frequency's embedding network
            freq_params = {
                'model': 'mts_sequential_forecast_lstm',
                'dynamic_inputs':
                    cfg.dynamic_inputs[freq] if isinstance(cfg.dynamic_inputs, dict) else cfg.dynamic_inputs,
                'hindcast_inputs':
                    cfg.hindcast_inputs[freq] if isinstance(cfg.hindcast_inputs, dict) else cfg.hindcast_inputs,
                'forecast_inputs':
                    cfg.forecast_inputs[freq] if isinstance(cfg.forecast_inputs, dict) else cfg.forecast_inputs,
                'dynamics_embedding': cfg.dynamics_embedding,
                'static_attributes': cfg.static_attributes,
                'statics_embedding': cfg.statics_embedding,
                'use_basin_id_encoding': cfg.use_basin_id_encoding,
                'number_of_basins': cfg.number_of_basins,
                'head': cfg.head,
                'seq_length': cfg.seq_length[freq] if isinstance(cfg.seq_length, dict) else cfg.seq_length
            }
            freq_cfg = Config(freq_params)

            # DEBUGGING
            self.hindcast_embedding_net[freq] = InputLayer(freq_cfg, embedding_type='hindcast')
            self.forecast_embedding_net[freq] = InputLayer(freq_cfg, embedding_type='forecast')
            # if freq == self._highest_freq:
            #     self.hindcast_embedding_net[freq] = InputLayer(freq_cfg, embedding_type='hindcast')
            #     self.forecast_embedding_net[freq] = InputLayer(freq_cfg, embedding_type='forecast')
            # else:
            #     self.hindcast_embedding_net[freq] = InputLayer(freq_cfg, embedding_type='full_model')

        if self.hindcast_embedding_net[self._highest_freq].output_size != self.forecast_embedding_net[
                self._highest_freq].output_size:
            raise ValueError('Forecast and hindcast embedding nets must have the same output size '
                             'for the highest frequency.')

        if not isinstance(cfg.hidden_size, dict):
            LOGGER.info("No specific hidden size for frequencies are specified. Same hidden size is used for all.")
            self._hidden_size = {freq: cfg.hidden_size for freq in self._frequencies}
        else:
            self._hidden_size = cfg.hidden_size

        if (self._is_shared_mtslstm or self._transfer_mtslstm_states["h"] == "identity"
                or self._transfer_mtslstm_states["c"] == "identity") and any(
                    size != self._hidden_size[self._frequencies[0]] for size in self._hidden_size.values()):
            raise ValueError("All hidden sizes must be equal if shared_mtslstm is used or state transfer=identity.")

        self._init_modules()
        self._reset_parameters()
        self._init_frequency_factors_and_slice_timesteps()

    def _init_modules(self):
        self.lstms = nn.ModuleDict()
        self.transfer_fcs = nn.ModuleDict()
        self.heads = nn.ModuleDict()
        self.dropout = nn.Dropout(p=self.cfg.output_dropout)

        for idx, freq in enumerate(self._frequencies):
            freq_input_size = self.hindcast_embedding_net[freq].output_size
            if self._is_shared_mtslstm:
                freq_input_size += len(self._frequencies)
            if self.cfg.head.lower() == "umal":
                freq_input_size += 1

            # if self._is_shared_mtslstm and idx > 0:
            #     self.lstms[freq] = self.lstms[self._frequencies[idx - 1]]
            #     self.heads[freq] = self.heads[self._frequencies[idx - 1]]
            # else:
            #     self.lstms[freq] = nn.LSTM(input_size=freq_input_size, hidden_size=self._hidden_size[freq])
            #     self.heads[freq] = get_head(self.cfg, n_in=self._hidden_size[freq], n_out=self.output_size)
            self.lstms[f'{freq}_hindcast'] = nn.LSTM(input_size=freq_input_size, hidden_size=self._hidden_size[freq])
            self.lstms[f'{freq}_forecast'] = nn.LSTM(input_size=freq_input_size, hidden_size=self._hidden_size[freq])
            self.heads[freq] = get_head(self.cfg, n_in=self._hidden_size[freq], n_out=self.output_size)

            if idx < len(self._frequencies) - 1: # Because we don't need a transfer function for the highest state
                for state in ["c", "h"]:
                    if self._transfer_mtslstm_states[state] == "linear":
                        self.transfer_fcs[f"{state}_{freq}"] = nn.Linear(self._hidden_size[freq],
                                                                         self._hidden_size[self._frequencies[idx + 1]])
                    elif self._transfer_mtslstm_states[state] == "identity":
                        self.transfer_fcs[f"{state}_{freq}"] = nn.Identity()
                    else:
                        pass

    def _init_frequency_factors_and_slice_timesteps(self):
        for idx, freq in enumerate(self._frequencies):
            if idx < len(self._frequencies) - 1:
                frequency_factor = get_frequency_factor(freq, self._frequencies[idx + 1])
                if frequency_factor != int(frequency_factor):
                    raise ValueError('Adjacent frequencies must be multiples of each other.')
                self._frequency_factors.append(int(frequency_factor))
                slice_timestep = int(self._seq_lengths[self._frequencies[idx + 1]] / self._frequency_factors[idx])
                self._slice_timestep[freq] = slice_timestep

    def _reset_parameters(self):
        if self.cfg.initial_forget_bias is not None:
            for freq in self._frequencies:
                hidden_size = self._hidden_size[freq]
                self.lstms[f'{freq}_hindcast'].bias_hh_l0.data[hidden_size:2 * hidden_size] = self.cfg.initial_forget_bias
                self.lstms[f'{freq}_forecast'].bias_hh_l0.data[hidden_size:2 * hidden_size] = self.cfg.initial_forget_bias

    def _add_frequency_one_hot_encoding(self, x_d: torch.Tensor, freq: str) -> torch.Tensor:
        idx = self._frequencies.index(freq)
        one_hot_freq = torch.zeros(x_d.shape[0], x_d.shape[1], len(self._frequencies), device=x_d.device)
        one_hot_freq[:, :, idx] = 1
        return torch.cat([x_d, one_hot_freq], dim=2)

    def _prepare_inputs(self, data: Dict[str, torch.Tensor], freq: str, embedding_type: str) -> torch.Tensor:
        """Concat all different inputs to the time series input"""
        
        if self.hindcast_embedding_net is not None: # FIXME should test for hindcast_embedding_net[freq] and forecast_embedding_net[freq]
            # use embedding network if available
            input_data = {
                f'x_d_{embedding_type}': data[f'x_d_{freq}_{embedding_type}'],
                'x_s': data.get('x_s'),
                'x_one_hot': data.get('x_one_hot')
            }
            # if specific x_s for frequency is available, use that
            if f'x_s_{freq}' in data:
                input_data['x_s'] = data[f'x_s_{freq}']
                
            # filter out None values
            input_data = {k: v for k, v in input_data.items() if v is not None}

            if embedding_type == 'hindcast':
                x_d = self.hindcast_embedding_net[freq](input_data, concatenate_output=True)
            elif embedding_type == 'forecast':
                x_d = self.forecast_embedding_net[freq](input_data, concatenate_output=True)
            # FIXME - check whether we need to implement this           
            # # add frequency one-hot encoding if shared_mtslstm is used
            # if self._is_shared_mtslstm:
            #     x_d = self._add_frequency_one_hot_encoding(x_d, freq)

        else:
            # directly use the input data without embedding, following the original implementation
            suffix = f"_{freq}"
            
            # concatenate all dynamic feature tensors from the dictionary
            feature_tensors = []
            for feature_name, feature_tensor in data[f'x_d{suffix}_{embedding_type}'].items():
                feature_tensors.append(feature_tensor)
            
            if feature_tensors:
                # concatenate all features along the feature dimension
                x_d = torch.cat(feature_tensors, dim=-1).transpose(0, 1)
            else:
                # no dynamic features found, which is invalid
                raise ValueError(f"No dynamic features found for frequency {freq}.")

            # concat all static and one-hot encoded features
            if f'x_s{suffix}' in data and 'x_one_hot' in data:
                x_s = data[f'x_s{suffix}'].unsqueeze(0).repeat(x_d.shape[0], 1, 1)
                x_one_hot = data['x_one_hot'].unsqueeze(0).repeat(x_d.shape[0], 1, 1)
                x_d = torch.cat([x_d, x_s, x_one_hot], dim=-1)
            elif f'x_s{suffix}' in data:
                x_s = data[f'x_s{suffix}'].unsqueeze(0).repeat(x_d.shape[0], 1, 1)
                x_d = torch.cat([x_d, x_s], dim=-1)
            elif 'x_s' in data:
                x_s = data['x_s'].unsqueeze(0).repeat(x_d.shape[0], 1, 1)
                x_d = torch.cat([x_d, x_s], dim=-1)
            elif 'x_one_hot' in data:
                x_one_hot = data['x_one_hot'].unsqueeze(0).repeat(x_d.shape[0], 1, 1)
                x_d = torch.cat([x_d, x_one_hot], dim=-1)
            else:
                pass

            if self._is_shared_mtslstm:
                x_d = self._add_frequency_one_hot_encoding(x_d, freq)
        
        return x_d

    def forward(self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:

        x_d_hindcast = {freq: self._prepare_inputs(data, freq, 'hindcast') for freq in self._frequencies}
        x_d_forecast = {freq: self._prepare_inputs(data, freq, 'forecast') for freq in self._frequencies}

        # x_f = self.forecast_embedding_net[self._highest_freq](data)
        # if self._is_shared_mtslstm:
        #     x_f = self._add_frequency_one_hot_encoding(x_f, self._highest_freq)
        # x_d_forecast = {self._highest_freq: x_f}

        # Initial states for lowest frequencies are set to zeros
        batch_size = x_d_hindcast[self._frequencies[0]].shape[1]
        lowest_freq_hidden_size = self._hidden_size[self._frequencies[0]]
        h_0_transfer = x_d_hindcast[self._frequencies[0]].new_zeros((1, batch_size, lowest_freq_hidden_size))
        c_0_transfer = torch.zeros_like(h_0_transfer)

        outputs = {}
        for idx, freq in enumerate(self._frequencies):
            if idx < len(self._frequencies) - 1: # lower frequency

                x_h = x_d_hindcast[freq]
                x_f = x_d_forecast[freq]

                slice_timestep = self._slice_timestep[freq]
                slice_timestep -= len(x_f)

                # Hindcast to transition from low resolution to high resolution
                lstm_output_hindcast1, (h_n_hindcast1, c_n_hindcast1) = self.lstms[f'{freq}_hindcast'](x_h[:-slice_timestep], (h_0_transfer, c_0_transfer))

                # Pass the hidden and cell states through the transfer function to supply to higher resolution LSTM
                if self._transfer_mtslstm_states["h"] is not None:
                    h_0_transfer = self.transfer_fcs[f"h_{freq}"](h_n_hindcast1)
                if self._transfer_mtslstm_states["c"] is not None:
                    c_0_transfer = self.transfer_fcs[f"c_{freq}"](c_n_hindcast1)

                # Same LSTM is used before and after state transfer — time meaning has not changed.
                # We simply pass state to the finer timescale, then continue the same-frequency modeling.
                lstm_output_hindcast2, (h_n_hindcast2, c_n_hindcast2) = self.lstms[f'{freq}_hindcast'](x_h[-slice_timestep:], (h_n_hindcast1, c_n_hindcast1))

                # Supply hidden and cell states to forecast data
                lstm_output_forecast, _ = self.lstms[f'{freq}_forecast'](x_f, (h_n_hindcast2, c_n_hindcast2)) 

                # Run head and update outputs
                lstm_output = torch.cat([lstm_output_hindcast1, lstm_output_hindcast2, lstm_output_forecast], dim=0)
                lstm_output = lstm_output.transpose(0, 1)
                head_out = self.heads[freq](self.dropout(lstm_output)) #.transpose(0, 1)))
                outputs.update({f'{key}_{freq}': value for key, value in head_out.items()})

                # print(f'{freq}: {lstm_output_hindcast1.shape}')
                # print(f'{freq}: {lstm_output_hindcast2.shape}')
                # print(f'{freq}: {lstm_output_forecast.shape}')
                # print(f'{freq}: {lstm_output.shape}')
                # for key, value in outputs.items(): 
                #     print(f'{key} shape: {value.shape}')

            else:  # highest frequency

                x_h = x_d_hindcast[freq]
                x_f = x_d_forecast[freq]

                # Hindcast portion of highest resolution data - this overlaps with `lstm_output_hindcast2`, but at the higher resolution
                lstm_output_hindcast, (h_n_hindcast, c_n_hindcast) = self.lstms[f'{freq}_hindcast'](x_h, (h_0_transfer, c_0_transfer))

                # Forecast portion of highest resolution data 
                lstm_output_forecast, (h_n_forecast, c_n_forecast) = self.lstms[f'{freq}_forecast'](x_f, (h_n_hindcast, c_n_hindcast))

                lstm_output = torch.cat([lstm_output_hindcast, lstm_output_forecast], dim=0)
                lstm_output = lstm_output.transpose(0, 1)
                head_out = self.heads[freq](self.dropout(lstm_output))
                outputs.update({f'{key}_{freq}': value for key, value in head_out.items()})

                h_n_hindcast = h_n_hindcast.transpose(0, 1)
                c_n_hindcast = c_n_hindcast.transpose(0, 1)
                h_n_forecast = h_n_forecast.transpose(0, 1)
                c_n_forecast = c_n_forecast.transpose(0, 1)

                # print(f'{freq}: {lstm_output_hindcast.shape}')
                # print(f'{freq}: {lstm_output_forecast.shape}')
                # print(f'{freq}: {lstm_output.shape}')

                outputs.update({
                    f'lstm_output_hindcast_{freq}': lstm_output_hindcast,
                    f'lstm_output_forecast_{freq}': lstm_output_forecast,
                    f'h_n_hindcast_{freq}': h_n_hindcast,
                    f'c_n_hindcast_{freq}': c_n_hindcast,
                    f'h_n_forecast_{freq}': h_n_forecast,
                    f'c_n_forecast_{freq}': c_n_forecast,
                })

                # for key, value in outputs.items(): 
                #     print(f'{key} shape: {value.shape}')

        return outputs
