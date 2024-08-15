import logging
import pickle
# import re
import sys
import warnings
# from collections import defaultdict
from typing import List, Dict, Union
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
# from pandas.tseries import frequencies
from pandas.tseries.frequencies import to_offset
import torch
import xarray
from numba import NumbaPendingDeprecationWarning
from numba import njit, prange
# from ruamel.yaml import YAML
# from torch.utils.data import Dataset
from tqdm import tqdm

# from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.datasetzoo.genericdataset import GenericDataset
from neuralhydrology.datautils import utils
from neuralhydrology.utils.config import Config
from neuralhydrology.utils.errors import NoTrainDataError, NoEvaluationDataError
# from neuralhydrology.utils import samplingutils


LOGGER = logging.getLogger(__name__)


class ForecastDataset(GenericDataset):
    """Data set class for the generic dataset that reads data for any region based on common file layout conventions.

    To use this dataset, the data_dir must contain a folder 'time_series' and (if static attributes are used) a folder
    'attributes'. The folder 'time_series' contains one netcdf file (.nc or .nc4) per basin, named '<basin_id>.nc/nc4'.
    The netcdf file has to have one coordinate called `date`, containing the datetime index. The folder 'attributes' 
    contains one or more comma-separated file (.csv) with static attributes, indexed by basin id. Attributes files can 
    be divided into groups of basins or groups of features (but not both, see `genericdataset.load_attributes` for
    more details).

    Note: Invalid values have to be marked as NaN (e.g. using NumPy's np.nan) in the netCDF files and not something like
    -999 for invalid discharge measurements, which is often found in hydrology datasets. If missing values are not 
    marked as NaN's, the GenericDataset will not be able to identify these values as missing data points.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    is_train : bool
        Defines if the dataset is used for training or evaluating. If True (training), means/stds for each feature
        are computed and stored to the run directory. If one-hot encoding is used, the mapping for the one-hot encoding
        is created and also stored to disk. If False, a `scaler` input is expected and similarly the `id_to_int` input
        if one-hot encoding is used.
    period : {'train', 'validation', 'test'}
        Defines the period for which the data will be loaded
    basin : str, optional
        If passed, the data for only this basin will be loaded. Otherwise the basin(s) are read from the appropriate
        basin file, corresponding to the `period`.
    additional_features : List[Dict[str, pd.DataFrame]], optional
        List of dictionaries, mapping from a basin id to a pandas DataFrame. This DataFrame will be added to the data
        loaded from the dataset and all columns are available as 'dynamic_inputs', 'evolving_attributes' and
        'target_variables'
    id_to_int : Dict[str, int], optional
        If the config argument 'use_basin_id_encoding' is True in the config and period is either 'validation' or
        'test', this input is required. It is a dictionary, mapping from basin id to an integer (the one-hot encoding).
    scaler : Dict[str, Union[pd.Series, xarray.DataArray]], optional
        If period is either 'validation' or 'test', this input is required. It contains the centering and scaling
        for each feature and is stored to the run directory during training (train_data/train_data_scaler.yml).
    """

    def __init__(self,
                 cfg: Config,
                 is_train: bool,
                 period: str,
                 basin: str = None,
                 additional_features: List[Dict[str, pd.DataFrame]] = [],
                 id_to_int: Dict[str, int] = {},
                 scaler: Dict[str, Union[pd.Series, xarray.DataArray]] = {}):
    
        super(GenericDataset, self).__init__(cfg=cfg,
                                             is_train=is_train,
                                             period=period,
                                             basin=basin,
                                             additional_features=additional_features,
                                             id_to_int=id_to_int,
                                             scaler=scaler)

    def _load_basin_data(self, basin: str, columns: list) -> pd.DataFrame:
        """Load input and output data. """
        # modified so that we can specify the columns to load - this allows us to select hindcast/forecast variables separately. 
        df = load_timeseries(data_dir=self.cfg.data_dir, time_series_data_sub_dir=self.cfg.time_series_data_sub_dir, basin=basin, columns=columns)
        return df

    def _initialize_frequency_configuration(self):
        """Checks and extracts configuration values for 'use_frequency', 'seq_length', and 'predict_last_n'"""

        self.seq_len = self.cfg.seq_length
        self._forecast_seq_len = self.cfg.forecast_seq_length
        self._predict_last_n = self.cfg.predict_last_n
        self._forecast_offset = self.cfg.forecast_offset

        # NOTE this dataset does not currently support multiple timestep frequencies. Instead 
        # we populate use_frequencies with the native frequency of the input data. 
        if self.cfg.use_frequencies:
            LOGGER.warning('Multiple timestep frequencies are not supported by this dataset: '
                           'defaulting to native frequency of input data')
        self.frequencies = []

        if not self.frequencies:
            if not isinstance(self.seq_len, int) or not isinstance(self._predict_last_n, int):
                raise ValueError('seq_length and predict_last_n must be integers')
            self.seq_len = [self.seq_len]
            self._forecast_seq_len = [self._forecast_seq_len]
            self._predict_last_n = [self._predict_last_n]

    def _load_or_create_xarray_dataset(self) -> xarray.Dataset:
        # if no netCDF file is passed, data set is created from raw basin files
        if (self.cfg.train_data_file is None) or (not self.is_train):
            data_list = []
            
            # not supported: self.cfg.evolving_attributes, self.cfg.mass_inputs, self.cfg.autoregressive_inputs
            hcst_keep_cols = self.cfg.target_variables + self.cfg.hindcast_inputs
            fcst_keep_cols = self.cfg.forecast_inputs 

            if not self._disable_pbar:
                LOGGER.info("Loading basin data into xarray data set.")
            for basin in tqdm(self.basins, disable=self._disable_pbar, file=sys.stdout):

                df_hcst = self._load_basin_data(basin, hcst_keep_cols)
                df_fcst = self._load_basin_data(basin, fcst_keep_cols) 

                # Make sure the multiindex is ordered correctly
                df_fcst.reset_index(inplace=True)
                df_fcst.set_index(['date', 'lead_time'], inplace=True)
                lead_times = df_fcst.index.unique(level='lead_time')

                # add columns from dataframes passed as additional data files
                df_hcst = pd.concat([df_hcst, *[d[basin] for d in self.additional_features]], axis=1)
                df_fcst = pd.concat([df_fcst, *[d[basin] for d in self.additional_features]], axis=1)
                # if target variables are missing for basin, add empty column to still allow predictions to be made
                if not self.is_train:
                    # target variables held in hcst dataset
                    df_hcst = self._add_missing_targets(df_hcst)

                # check if any feature should be duplicated
                df_hcst = self._duplicate_features(df_hcst)
                df_fcst = self._duplicate_features(df_fcst)

                # check if a shifted copy of a feature should be added
                df_fcst = self._add_lagged_features(df_fcst)
                df_hcst = self._add_lagged_features(df_hcst)

                # TODO do this in a consistent way, so that hindcast/forecast data have the same missing portions
                # # remove random portions of the timeseries of dynamic features
                # for holdout_variable, holdout_dict in self.cfg.random_holdout_from_dynamic_features.items():
                #     df[holdout_variable] = samplingutils.bernoulli_subseries_sampler(
                #         data=df[holdout_variable].values,
                #         missing_fraction=holdout_dict['missing_fraction'],
                #         mean_missing_length=holdout_dict['mean_missing_length'],
                #     )

                # Make end_date the last second of the specified day, such that the
                # dataset will include all hours of the last day, not just 00:00.
                start_dates = self.start_and_end_dates[basin]["start_dates"]
                end_dates = [
                    date + pd.Timedelta(days=1, seconds=-1) for date in self.start_and_end_dates[basin]["end_dates"]
                ]

                # infer native frequency from hindcast data 
                native_frequency = utils.infer_frequency(df_hcst.index)
                if not self.frequencies:
                    self.frequencies = [native_frequency]  # use df's native resolution by default

                # Assert that the used frequencies are lower or equal than the native frequency. There may be cases
                # where our logic cannot determine whether this is the case, because pandas might return an exotic
                # native frequency. In this case, all we can do is print a warning and let the user check themselves.
                try:
                    freq_vs_native = [utils.compare_frequencies(freq, native_frequency) for freq in self.frequencies]
                except ValueError:
                    LOGGER.warning('Cannot compare provided frequencies with native frequency. '
                                   'Make sure the frequencies are not higher than the native frequency.')
                    freq_vs_native = []
                if any(comparison > 1 for comparison in freq_vs_native):
                    raise ValueError(f'Frequency is higher than native data frequency {native_frequency}.')

                # used to get the maximum warmup-offset across all frequencies. We don't use to_timedelta because it
                # does not support all frequency strings. We can't calculate the maximum offset here, because to
                # compare offsets, they need to be anchored to a specific date (here, the start date).
                # NOTE in this dataset, `offsets` is the number of hindcast timesteps that need to be included in the warmup
                offsets = [(self.seq_len[i] - max(self._predict_last_n[i], self._forecast_seq_len[i])) * to_offset(freq)
                           for i, freq in enumerate(self.frequencies)]
                
                hcst_basin_data_list = []
                fcst_basin_data_list = []
                
                # create xarray data set for each period slice of the specific basin
                for i, (start_date, end_date) in enumerate(zip(start_dates, end_dates)):
                    # if the start date is not aligned with the frequency, the resulting datetime indices will be off
                    if not all(to_offset(freq).is_on_offset(start_date) for freq in self.frequencies):
                        misaligned = [freq for freq in self.frequencies if not to_offset(freq).is_on_offset(start_date)]
                        raise ValueError(f'start date {start_date} is not aligned with frequencies {misaligned}.')

                    # add warmup period, so that we can make prediction at the first time step specified by period.
                    # offsets has the warmup offset needed for each frequency; the overall warmup starts with the
                    # earliest date, i.e., the largest offset across all frequencies.
                    warmup_start_date = min(start_date - offset for offset in offsets)
                    df_hcst_sub = df_hcst[warmup_start_date:end_date]

                    # `df_fcst_sub` has a multiindex
                    idx = pd.IndexSlice
                    df_fcst_sub = df_fcst.loc[idx[start_date:end_date, :], :]

                    # make sure the df covers the full date range from warmup_start_date to end_date, filling any gaps
                    # with NaNs. This may increase runtime, but is a very robust way to make sure dates and predictions
                    # keep in sync. In training, the introduced NaNs will be discarded, so this only affects evaluation.
                    full_range = pd.date_range(start=warmup_start_date, end=end_date, freq=native_frequency)
                    df_hcst_sub = df_hcst_sub.reindex(pd.DatetimeIndex(full_range, name=df_hcst_sub.index.name))
                    
                    # Select the rows between the start_date and end_date
                    # TODO specify lead_times in config
                    full_range = pd.date_range(start=start_date, end=end_date, freq=native_frequency)
                    df_fcst_sub = df_fcst_sub.reindex(pd.MultiIndex.from_product([full_range, lead_times], names=['date', 'lead_time']))

                    # Set all targets before period start to NaN
                    df_hcst_sub.loc[df_hcst_sub.index < start_date, self.cfg.target_variables] = np.nan

                    hcst_basin_data_list.append(df_hcst_sub)
                    fcst_basin_data_list.append(df_fcst_sub)

                if not hcst_basin_data_list:
                    # Skip basin in case no start and end dates where defined.
                    continue

                # In case of multiple time slices per basin, stack the time slices in the time dimension.
                df_hcst = pd.concat(hcst_basin_data_list, axis=0)
                df_fcst = pd.concat(fcst_basin_data_list, axis=0)

                # Because of overlaps between warmup period of one slice and training period of another slice, there can
                # be duplicated indices. The next block of code creates two subset dataframes. First, a subset with all
                # non-duplicated indices. Second, a subset with duplicated indices, of which we keep the rows, where the
                # target value is not NaN (because we remove the target variable during warmup periods but want to keep
                # them if they are target in another temporal slice).
                # NOTE this is not a problem for `df_fcst` because no warmup period is needed, so we only check `df_hcst`
                df_non_duplicated = df_hcst[~df_hcst.index.duplicated(keep=False)]
                df_duplicated = df_hcst[df_hcst.index.duplicated(keep=False)]
                filtered_duplicates = []
                for _, grp in df_duplicated.groupby('date'):
                    mask = ~grp[self.cfg.target_variables].isna().any(axis=1)
                    if not mask.any():
                        # In case all duplicates have a NaN value for the targets, pick the first. This can happen, if
                        # the day itself has a missing observation.
                        filtered_duplicates.append(grp.head(1))
                    else:
                        # If at least one duplicate has values in the target columns, take the first of these rows.
                        filtered_duplicates.append(grp[mask].head(1))

                if filtered_duplicates:
                    # Combine the filtered duplicates with the non-duplicates.
                    df_filtered_duplicates = pd.concat(filtered_duplicates, axis=0)
                    df_hcst = pd.concat([df_non_duplicated, df_filtered_duplicates], axis=0)
                else:
                    # Else, if no duplicates existed, continue with only the non-duplicate df.
                    df_hcst = df_non_duplicated

                # Sort by DatetimeIndex and reindex to fill gaps with NaNs.
                df_hcst = df_hcst.sort_index(axis=0, ascending=True)
                df_hcst = df_hcst.reindex(
                    pd.DatetimeIndex(data=pd.date_range(df_hcst.index[0], df_hcst.index[-1], freq=native_frequency),
                                     name=df_hcst.index.name))
                
                # Convert to xarray Dataset and add basin string as additional coordinate
                xr_fcst = xarray.Dataset.from_dataframe(df_fcst.astype(np.float32))
                xr_hcst = xarray.Dataset.from_dataframe(df_hcst.astype(np.float32))

                # merging xarray datasets has the convenient side-effect that both forecast and hindcast data will have the same temporal extent
                xr = xr_fcst.merge(xr_hcst) 
                xr = xr.assign_coords({'basin': basin})
                data_list.append(xr)

            if not data_list:
                # If no period for no basin has defined timeslices, raise error.
                if self.is_train:
                    raise NoTrainDataError
                else:
                    raise NoEvaluationDataError

            # create one large dataset that has two coordinates: datetime and basin
            xr = xarray.concat(data_list, dim="basin")
            if self.is_train and self.cfg.save_train_data:
                self._save_xarray_dataset(xr)

        else:
            # Otherwise we can reload previously-saved training data
            with self.cfg.train_data_file.open("rb") as fp:
                d = pickle.load(fp)
            xr = xarray.Dataset.from_dict(d)
            if not self.frequencies:
                native_frequency = utils.infer_frequency(xr["date"].values)
                self.frequencies = [native_frequency]

        return xr

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        basin, indices = self.lookup_table[item]

        sample = {}
        for freq, seq_len, forecast_seq_len, idx in zip(self.frequencies, self.seq_len, self._forecast_seq_len, indices):
            # if there's just one frequency, don't use suffixes.
            freq_suffix = '' if len(self.frequencies) == 1 else f'_{freq}'
            
            # NOTE idx is the index of the forecast issue time
            # hence, idx + self._forecast_offset is the index of the first forecast
            hindcast_start_idx = idx + self._forecast_offset + forecast_seq_len - seq_len
            hindcast_end_idx = idx + self._forecast_offset # slice-end is excluding - we take values up, but not including, the issue time
            # `forecast_start_idx` equals `idx` because in the case of forecasts, the 
            # time dimension refers to the initialization time and NOT the time of the 
            # forecast itself, which instead is indexed by lead time. 
            forecast_start_idx = idx
            global_end_idx = idx + self._forecast_offset + forecast_seq_len 

            # TODO allow forecast overlap - currently not supported
            # hindcast_end_idx = idx + 1 - self.cfg.forecast_seq_length
            # forecast_start_idx = idx + 1 - self.cfg.forecast_seq_length
            # if self.cfg.forecast_overlap and self.cfg.forecast_overlap > 0:
            #     hindcast_end_idx += self.cfg.forecast_overlap
            sample[f'x_h{freq_suffix}'] = self._x_h[basin][freq][hindcast_start_idx:hindcast_end_idx]
            sample[f'x_f{freq_suffix}'] = self._x_f[basin][freq][forecast_start_idx]
            sample[f'y{freq_suffix}'] = self._y[basin][freq][hindcast_start_idx:global_end_idx]
            sample[f'date{freq_suffix}'] = self._dates[basin][freq][hindcast_start_idx:global_end_idx]

            # check for static inputs
            static_inputs = []
            if self._attributes:
                static_inputs.append(self._attributes[basin])
            if self._x_s:
                static_inputs.append(self._x_s[basin][freq][idx])
            if static_inputs:
                sample[f'x_s{freq_suffix}'] = torch.cat(static_inputs, dim=-1)

            if self.cfg.timestep_counter:
                torch.concatenate([sample[f'x_h{freq_suffix}'], self.hindcast_counter], dim=-1)
                torch.concatenate([sample[f'x_f{freq_suffix}'], self.forecast_counter], dim=-1)

        if self._per_basin_target_stds:
            sample['per_basin_target_stds'] = self._per_basin_target_stds[basin]
        if self.id_to_int:
            sample['x_one_hot'] = torch.nn.functional.one_hot(torch.tensor(self.id_to_int[basin]),
                                                              num_classes=len(self.id_to_int)).to(torch.float32)

        return sample
    
    def _create_lookup_table(self, xr: xarray.Dataset):
        lookup = []
        if not self._disable_pbar:
            LOGGER.info("Create lookup table and convert to pytorch tensor")

        # Split data into forecast and hindcast components
        xr_fcst = xr[self.cfg.forecast_inputs]
        xr_hcst = xr[[var for var in xr.variables if var not in self.cfg.forecast_inputs]]
        xr_hcst = xr_hcst.drop_dims('lead_time')

        # list to collect basins ids of basins without a single training sample
        basins_without_samples = []
        basin_coordinates = xr_hcst["basin"].values.tolist()
        for basin in tqdm(basin_coordinates, file=sys.stdout, disable=self._disable_pbar):
            # store data of each frequency as numpy array of shape [time steps, features] and dates as numpy array of
            # shape (time steps,)
            x_h, x_f, x_s, y, dates = {}, {}, {}, {}, {}

            # keys: frequencies, values: array mapping each lowest-frequency
            # sample to its corresponding sample in this frequency
            frequency_maps = {}
            lowest_freq = utils.sort_frequencies(self.frequencies)[0]

            # converting from xarray to pandas DataFrame because resampling is much faster in pandas.
            df_hcst_native = xr_hcst.sel(basin=basin, drop=True).to_dataframe()
            df_fcst_native = xr_fcst.sel(basin=basin, drop=True).to_dataframe()

            for freq in self.frequencies:

                # multiple frequencies are not supported so we don't do any resampling here
                df_hcst_resampled = df_hcst_native
                df_fcst_resampled = df_fcst_native
            
                # pull all of the data that needs to be validated
                x_h[freq] = df_hcst_resampled[self.cfg.hindcast_inputs].values
                # cast multiindex dataframe to three-dimensional array
                x_f[freq] = df_fcst_resampled[self.cfg.forecast_inputs].to_xarray().to_array().transpose('date', 'lead_time', 'variable').values
                y[freq] = df_hcst_resampled[self.cfg.target_variables].values
                
                # Add dates of the (resampled) data to the dates dict
                dates[freq] = df_hcst_resampled.index.to_numpy()

                # number of frequency steps in one lowest-frequency step
                frequency_factor = int(utils.get_frequency_factor(lowest_freq, freq))
                # array position i is the last entry of this frequency that belongs to the lowest-frequency sample i.
                if len(df_hcst_resampled) % frequency_factor != 0:
                    raise ValueError(f'The length of the dataframe at frequency {freq} is {len(df_resampled)} '
                                     f'(including warmup), which is not a multiple of {frequency_factor} (i.e., the '
                                     f'factor between the lowest frequency {lowest_freq} and the frequency {freq}. '
                                     f'To fix this, adjust the {self.period} start or end date such that the period '
                                     f'(including warmup) has a length that is divisible by {frequency_factor}.')
                frequency_maps[freq] = np.arange(len(df_hcst_resampled) // frequency_factor) \
                                       * frequency_factor + (frequency_factor - 1)


            # store first date of sequence to be able to restore dates during inference
            if not self.is_train:
                self.period_starts[basin] = pd.to_datetime(xr_hcst.sel(basin=basin)["date"].values[0])

            # we can ignore the deprecation warning about lists because we don't use the passed lists
            # after the validate_samples call. The alternative numba.typed.Lists is still experimental.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

                # checks inputs and outputs for each sequence. valid: flag = 1, invalid: flag = 0
                # manually unroll the dicts into lists to make sure the order of frequencies is consistent.
                # during inference, we want all samples with sufficient history (even if input is NaN), so
                # we pass x_d, x_s, y as None.
                flag = validate_samples(x_h=[x_h[freq] for freq in self.frequencies] if self.is_train else None,
                                        x_f=[x_f[freq] for freq in self.frequencies] if self.is_train else None,
                                        y=[y[freq] for freq in self.frequencies] if self.is_train else None,
                                        seq_length=self.seq_len,
                                        forecast_seq_length=self._forecast_seq_len,
                                        forecast_offset=self._forecast_offset,
                                        predict_last_n=self._predict_last_n,
                                        frequency_maps=[frequency_maps[freq] for freq in self.frequencies])

            # # Concatenate autoregressive columns to dynamic inputs *after* validation, so as to not remove
            # # samples with missing autoregressive inputs.
            # # AR inputs must go at the end of the df/array (this is assumed by the AR model).
            # if self.cfg.autoregressive_inputs:
            #     for freq in self.frequencies:
            #         x_d[freq] = np.concatenate([x_d[freq], df_resampled[self.cfg.autoregressive_inputs].values], axis=1)
            #     x_d_column_names += self.cfg.autoregressive_inputs

            valid_samples = np.argwhere(flag == 1)
            self.valid_samples = valid_samples
            for f in valid_samples:
                # store pointer to basin and the sample's index in each frequency
                lookup.append((basin, [frequency_maps[freq][int(f)] for freq in self.frequencies]))

            self.lookup = lookup 
            # only store data if this basin has at least one valid sample in the given period
            if valid_samples.size > 0:
                if not self.cfg.hindcast_inputs:
                    raise ValueError('Hindcast inputs must be provided if forecast inputs are provided.')
                self._x_h[basin] = {freq: torch.from_numpy(_x_h.astype(np.float32)) for freq, _x_h in x_h.items()}
                self._x_f[basin] = {freq: torch.from_numpy(_x_f.astype(np.float32)) for freq, _x_f in x_f.items()}
                self._y[basin] = {freq: torch.from_numpy(_y.astype(np.float32)) for freq, _y in y.items()}
                self._dates[basin] = dates
            else:
                basins_without_samples.append(basin)

        if basins_without_samples:
            LOGGER.info(
                f"These basins do not have a single valid sample in the {self.period} period: {basins_without_samples}")
        self.lookup_table = {i: elem for i, elem in enumerate(lookup)}
        self.num_samples = len(self.lookup_table)
        self.valid_samples = valid_samples
        if self.num_samples == 0:
            if self.is_train:
                raise NoTrainDataError
            else:
                raise NoEvaluationDataError


@njit()
def validate_samples(x_h: List[np.ndarray], 
                     x_f: List[np.ndarray],
                     y: List[np.ndarray], 
                     seq_length: List[int],
                     forecast_seq_length: List[int],
                     forecast_offset: int, #List[int],
                     predict_last_n: List[int], 
                     frequency_maps: List[np.ndarray]) -> np.ndarray:
    """Checks for invalid samples due to NaN or insufficient sequence length.

    Parameters
    ----------
    x_h : List[np.ndarray]
        List of dynamic hindcast input data; one entry per frequency
    x_f : List[np.ndarray]
        List of dynamic forecast input data; one entry per frequency
    y : List[np.ndarray]
        List of target values; one entry per frequency
    seq_length : List[int]
        List of sequence lengths; one entry per frequency
    forecast_seq_length : List[int]
        List of forecast sequence lengths; one entry per frequency
    predict_last_n: List[int]
        List of predict_last_n; one entry per frequency
    frequency_maps : List[np.ndarray]
        List of arrays mapping lowest-frequency samples to their corresponding last sample in each frequency;
        one list entry per frequency.

    Returns
    -------
    np.ndarray 
        Array has a value of 1 for valid samples and a value of 0 for invalid samples.
    """

    # number of samples is number of lowest-frequency samples (all maps have this length)
    n_samples = len(frequency_maps[0])

    # 1 denotes valid sample, 0 denotes invalid sample
    flag = np.ones(n_samples)
    for i in range(len(frequency_maps)):  # iterate through frequencies
        for j in prange(n_samples):  # iterate through lowest-frequency samples

            # The length of the hindcast period is the total sequence length minus the length of the forecast sequence
            hindcast_seq_length = seq_length[i] - forecast_seq_length[i] 

            # find the last sample in this frequency that belongs to the lowest-frequency step j
            last_sample_of_freq = frequency_maps[i][j]
            # check whether there is sufficient data available to create a valid sequence (regardless of NaN etc, which are checked in the following sections)
            if last_sample_of_freq < (hindcast_seq_length): # - 1):
                flag[j] = 0  # too early for this frequency's seq_length (not enough history)
                continue

            # add forecast_offset here, because it determines how many timesteps ahead we're going to be predicting (remember forecast_offset is the number of timesteps between initialization and the first forecast)
            if (last_sample_of_freq + forecast_offset + forecast_seq_length[i]) > n_samples:
                flag[j] = 0
                continue 

            # any NaN in the hindcast inputs makes the sample invalid
            if x_h is not None:
                # NOTE hindcast stops the day before the forecast starts, so don't need to slice end
                _x_h = x_h[i][last_sample_of_freq - hindcast_seq_length + 1:last_sample_of_freq]
                if np.any(np.isnan(_x_h)):
                    flag[j] = 0
                    continue

            # any NaN in the forecast inputs make the sample invalid 
            if x_f is not None:
                _x_f = x_f[i][last_sample_of_freq]
                if np.any(np.isnan(_x_f)):
                    flag[j] = 0
                    continue

            # all-NaN in the targets makes the sample invalid
            if y is not None:
                _y = y[i][last_sample_of_freq - predict_last_n[i] + 1:last_sample_of_freq + 1]
                if np.prod(np.array(_y.shape)) > 0 and np.all(np.isnan(_y)):
                    flag[j] = 0
                    continue

    return flag


def load_timeseries(data_dir: Path, time_series_data_sub_dir: str, basin: str, columns: list) -> pd.DataFrame:
    """Load time series data from netCDF files into pandas DataFrame.

    Parameters
    ----------
    data_dir : Path
        Path to the data directory. This folder must contain a folder called 'time_series' containing the time series
        data for each basin as a single time-indexed netCDF file called '<basin_id>.nc/nc4'.
    basin : str
        The basin identifier.

    Returns
    -------
    pd.DataFrame
        Time-indexed DataFrame containing the time series data as stored in the netCDF file.

    Raises
    ------
    FileNotFoundError
        If no netCDF file exists for the specified basin.
    ValueError
        If more than one netCDF file is found for the specified basin.
    """
    files_dir = data_dir / "time_series"
    # Allow time series data from different members
    if time_series_data_sub_dir is not None:
        files_dir = files_dir / time_series_data_sub_dir

    netcdf_files = list(files_dir.glob("*.nc4"))
    netcdf_files.extend(files_dir.glob("*.nc"))
    netcdf_file = [f for f in netcdf_files if f.stem == basin]
    if len(netcdf_file) == 0:
        raise FileNotFoundError(f"No netCDF file found for basin {basin} in {files_dir}")
    if len(netcdf_file) > 1:
        raise ValueError(f"Multiple netCDF files found for basin {basin} in {files_dir}")

    xr = xarray.open_dataset(netcdf_file[0])
    xr = xr[columns]
    return xr.to_dataframe()