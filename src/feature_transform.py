import pandas as pd
import numpy as np
from typing import Optional
from sklearn.preprocessing import KBinsDiscretizer

from config.loader import cfg
from common import (
    parse_species_to_vector,
    assign_functional_groups,
    apply_smart_upsample_with_features,
    get_season,
)
from configs import get_state_specie_pair


class BiomassFeatureTransform:
    """
    Fit/Transform feature pipeline:
    - fit(train_df_raw):
        - Compute grouping keys (State_Specie)
        - Upsample training set
        - Engineer deterministic features (species vectors, interactions)
        - Fit KBinsDiscretizer on NDVI, Height, and composite biomass
    - transform(df_raw):
        - Apply deterministic features
        - Apply fitted binning on NDVI, Height, and composite
        - Compute session/season keys
    """

    def __init__(self, logger=None):
        self.logger = logger
        self.ndvi_kbd: Optional[KBinsDiscretizer] = None
        self.height_kbd: Optional[KBinsDiscretizer] = None
        self.comp_kbd: Optional[KBinsDiscretizer] = None
        self.n_bins = int(getattr(cfg.features, 'biomass_composite_bins', 5))

    def _log(self, msg: str):
        if self.logger:
            self.logger.info(msg)

    def _deterministic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Species lower and vectors
        df = df.copy()
        df['Species'] = df['Species'].astype(str).str.lower()
        core_species = cfg.species_taxonomy.core_species
        for sp in core_species:
            df[f'Species_{sp}'] = 0.0
        species_vectors = df['Species'].apply(parse_species_to_vector)
        species_matrix = np.stack(species_vectors.values)
        for i, sp in enumerate(core_species):
            df[f'Species_{sp}'] = species_matrix[:, i]

        # Functional groups
        df = assign_functional_groups(df)

        # Region-aware strat key
        df['State_Specie'] = df.apply(get_state_specie_pair, axis=1)

        # Soft species probabilities + richness
        species_cols_all = [f'Species_{sp}' for sp in core_species if f'Species_{sp}' in df.columns]
        if len(species_cols_all) > 0:
            species_count_internal = df[species_cols_all].sum(axis=1).astype(float)
            species_count_internal = species_count_internal.replace(0.0, 1.0)
            for sp in core_species:
                col = f'Species_{sp}'
                if col in df.columns:
                    df[f'SpeciesProb_{sp}'] = df[col].astype(float) / species_count_internal
                else:
                    df[f'SpeciesProb_{sp}'] = 0.0
            if cfg.features.use_species_count_feature:
                richness_cols = [f'Species_{sp}' for sp in core_species if sp != 'clover' and f'Species_{sp}' in df.columns]
                if len(richness_cols) > 0:
                    df['Species_Count'] = df[richness_cols].sum(axis=1).astype(float)

        # Deterministic auxiliary features
        df['Height_Ave_cm_log'] = np.log1p(df['Height_Ave_cm'].astype(float))
        df['Interaction_Mul'] = df['Pre_GSHH_NDVI'].astype(float) * df['Height_Ave_cm_log']
        df['Interaction_Add'] = df['Pre_GSHH_NDVI'].astype(float) + df['Height_Ave_cm_log']

        return df

    def _fit_bins(self, df: pd.DataFrame):
        # Fit NDVI/Height bins when requested
        if cfg.features.use_bin_features:
            try:
                if 'Pre_GSHH_NDVI' in df.columns:
                    self.ndvi_kbd = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')
                    self.ndvi_kbd.fit(df['Pre_GSHH_NDVI'].astype(float).values.reshape(-1, 1))
                if 'Height_Ave_cm' in df.columns:
                    self.height_kbd = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')
                    self.height_kbd.fit(df['Height_Ave_cm'].astype(float).values.reshape(-1, 1))
            except Exception:
                # Leave discretizers as None to skip
                self.ndvi_kbd = None
                self.height_kbd = None

        # Fit composite bins
        try:
            wts = cfg.targets.official_weights
            tgt_cols = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
            comp = np.zeros(len(df), dtype=float)
            for col, wt in zip(tgt_cols, wts):
                if col in df.columns:
                    comp += wt * df[col].astype(float).values
            self.comp_kbd = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='quantile')
            self.comp_kbd.fit(comp.reshape(-1, 1))
        except Exception:
            self.comp_kbd = None

    def _apply_bins(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # NDVI/Height
        if cfg.features.use_bin_features:
            if cfg.features.bin_encoding == 'ordinal':
                if self.ndvi_kbd is not None and 'Pre_GSHH_NDVI' in df.columns:
                    ndvi_bins = self.ndvi_kbd.transform(df['Pre_GSHH_NDVI'].astype(float).values.reshape(-1, 1)).astype(int).ravel()
                    df['NDVI_Bin_Ordinal'] = ndvi_bins.astype(float)
                if self.height_kbd is not None and 'Height_Ave_cm' in df.columns:
                    h_bins = self.height_kbd.transform(df['Height_Ave_cm'].astype(float).values.reshape(-1, 1)).astype(int).ravel()
                    df['Height_Bin_Ordinal'] = h_bins.astype(float)
            elif cfg.features.bin_encoding == 'onehot':
                if self.ndvi_kbd is not None and 'Pre_GSHH_NDVI' in df.columns:
                    ndvi_bins = self.ndvi_kbd.transform(df['Pre_GSHH_NDVI'].astype(float).values.reshape(-1, 1)).astype(int).ravel()
                    for k in range(4):
                        df[f'NDVI_Bin_OH_{k}'] = (ndvi_bins == k).astype(float)
                if self.height_kbd is not None and 'Height_Ave_cm' in df.columns:
                    h_bins = self.height_kbd.transform(df['Height_Ave_cm'].astype(float).values.reshape(-1, 1)).astype(int).ravel()
                    for k in range(4):
                        df[f'Height_Bin_OH_{k}'] = (h_bins == k).astype(float)

        # Composite
        if self.comp_kbd is not None:
            wts = cfg.targets.official_weights
            tgt_cols = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
            comp = np.zeros(len(df), dtype=float)
            for col, wt in zip(tgt_cols, wts):
                if col in df.columns:
                    comp += wt * df[col].astype(float).values
            bins = self.comp_kbd.transform(comp.reshape(-1, 1)).astype(int).ravel()
            df['biomass_binned_composite'] = bins
        else:
            df['biomass_binned_composite'] = 0

        return df

    def fit(self, train_df_raw: pd.DataFrame) -> pd.DataFrame:
        """Fit on TRAIN split. Returns engineered TRAIN df (after upsample)."""
        self._log("[Transform] Fitting on train split (with upsampling)")
        # Deterministic keys needed for upsample
        train_df_raw = train_df_raw.copy()
        train_df_raw['State_Specie'] = train_df_raw.apply(get_state_specie_pair, axis=1)
        # Upsample
        train_up = apply_smart_upsample_with_features(train_df_raw, self.logger)
        # Deterministic features on upsampled
        train_eng = self._deterministic_features(train_up)
        # Fit binning on upsampled train
        self._fit_bins(train_eng)
        # Apply bins to train
        train_final = self._apply_bins(train_eng)

        # Session/Season keys
        train_final['SessionID'] = train_final.apply(lambda r: f"{r['State']}_{pd.to_datetime(r['Sampling_Date']).strftime('%Y%m%d')}", axis=1)
        train_final['Season'] = train_final['Sampling_Date'].apply(get_season)
        train_final['Seasion_State_Specie'] = train_final.apply(lambda r: f"{r['Season']}_{r['State_Specie']}", axis=1)
        train_final['State_Season'] = train_final.apply(lambda r: f"{r['State']}_{r['Season']}", axis=1)
        train_final['Species_Season'] = train_final.apply(lambda r: f"{r['Species']}_{r['Season']}", axis=1)

        return train_final.reset_index(drop=True)

    def transform(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Transform VALIDATION or HOLDOUT splits using learned parameters."""
        self._log("[Transform] Applying to validation/holdout splits")
        df_eng = self._deterministic_features(df_raw)
        df_final = self._apply_bins(df_eng)
        df_final['SessionID'] = df_final.apply(lambda r: f"{r['State']}_{pd.to_datetime(r['Sampling_Date']).strftime('%Y%m%d')}", axis=1)
        df_final['Season'] = df_final['Sampling_Date'].apply(get_season)
        df_final['Season_State_Specie'] = df_final.apply(lambda r: f"{r['Season']}_{r['State_Specie']}", axis=1)
        df_final['State_Season'] = df_final.apply(lambda r: f"{r['State']}_{r['Season']}", axis=1)
        df_final['Species_Season'] = df_final.apply(lambda r: f"{r['Species']}_{r['Season']}", axis=1)
        return df_final.reset_index(drop=True)


def ensure_split_keys(df: pd.DataFrame, groupby_key: Optional[str], strat_key: Optional[str], logger=None) -> pd.DataFrame:
    """
    Deterministically derive keys needed for splitting on the raw dev set.
    - Does NOT upsample or learn from train; safe to run pre-split.
    - Computes common keys like 'State_Specie' and provisional composite bins.
    """
    d = df.copy()
    # Group key derivations
    if groupby_key == 'State_Specie' and 'State_Specie' not in d.columns:
        d['State_Specie'] = d.apply(get_state_specie_pair, axis=1)
    if groupby_key == 'GroupKey' and 'GroupKey' not in d.columns:
        d['DateStr'] = pd.to_datetime(d['Sampling_Date']).dt.strftime('%Y-%m-%d')
        species_col = 'species_id' if 'species_id' in d.columns else ('Species' if 'Species' in d.columns else None)
        if species_col is None:
            raise ValueError("No species column found (expected 'species_id' or 'Species').")
        d['GroupKey'] = d['State'].astype(str) + '|' + d[species_col].astype(str) + '|' + d['DateStr']

    # Provisional strat bins for splitting
    if strat_key == 'biomass_binned_composite' and 'biomass_binned_composite' not in d.columns:
        try:
            wts = cfg.targets.official_weights
            tgt_cols = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
            comp = np.zeros(len(d), dtype=float)
            for col, wt in zip(tgt_cols, wts):
                if col in d.columns:
                    comp += wt * d[col].astype(float).values
            n_bins = int(getattr(cfg.features, 'biomass_composite_bins', 5))
            kbd = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
            d['biomass_binned_composite'] = kbd.fit_transform(comp.reshape(-1, 1)).astype(int).ravel()
        except Exception:
            d['biomass_binned_composite'] = 0

    return d
