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
from configs import get_key1_specie_pair


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
        df['State_Specie'] = df.apply(get_key1_specie_pair, axis=1)

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
        
        # Session/Season keys (purely deterministic groupings)
        df['SessionID'] = df.apply(lambda r: f"{r['State']}_{pd.to_datetime(r['Sampling_Date']).strftime('%Y%m%d')}", axis=1)
        df['Season'] = df['Sampling_Date'].apply(get_season)
        df['Season_State_Specie'] = df.apply(lambda r: f"{r['Season']}_{r['State_Specie']}", axis=1)
        df['State_Season'] = df.apply(lambda r: f"{r['State']}_{r['Season']}", axis=1)
        df['Species_Season'] = df.apply(lambda r: f"{r['Species']}_{r['Season']}", axis=1)
        df["State_Sampling_Date"] = df.apply(lambda r: f"{r['State']}_{r['Sampling_Date']}", axis=1)

        return df

    def _fit_bins(self, df: pd.DataFrame):
        # Fit NDVI/Height bins when requested
        if cfg.features.use_bin_features:
            try:
                if 'Pre_GSHH_NDVI' in df.columns:
                    self.ndvi_kbd = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile', quantile_method='averaged_inverted_cdf')
                    self.ndvi_kbd.fit(df['Pre_GSHH_NDVI'].astype(float).values.reshape(-1, 1))
                if 'Height_Ave_cm' in df.columns:
                    self.height_kbd = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile', quantile_method='averaged_inverted_cdf')
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
            self.comp_kbd = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='quantile', quantile_method='averaged_inverted_cdf')
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
        # Expect deterministic features already applied pre-split
        train_up = apply_smart_upsample_with_features(train_df_raw.copy(), self.logger)
        train_eng = train_up
        # Fit binning on upsampled train
        self._fit_bins(train_eng)
        # Apply bins to train
        train_final = self._apply_bins(train_eng)

        return train_final.reset_index(drop=True)

    def transform(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Transform VALIDATION or HOLDOUT splits using learned parameters."""
        self._log("[Transform] Applying to validation/holdout splits")
        # Expect deterministic features already present
        df_final = self._apply_bins(df_raw)
        return df_final.reset_index(drop=True)


def apply_deterministic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Public helper to compute deterministic, non-learned features once pre-split."""
    ft = BiomassFeatureTransform(logger=None)
    df_det = ft._deterministic_features(df)

    # Create a global composite biomass bin column for use as a stratification key
    try:
        wts = cfg.targets.official_weights
        tgt_cols = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
        comp = np.zeros(len(df_det), dtype=float)
        for col, wt in zip(tgt_cols, wts):
            if col in df_det.columns:
                comp += wt * df_det[col].astype(float).values
        n_bins = int(getattr(cfg.features, 'biomass_composite_bins', 5))
        kbd = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile', quantile_method='averaged_inverted_cdf')
        bins = kbd.fit_transform(comp.reshape(-1, 1)).astype(int).ravel()
        # Add both the canonical 'biomass_composite_bins' name (used in config) and
        # the older 'biomass_binned_composite' for backward compatibility.
        df_det['biomass_composite_bins'] = bins
        df_det['biomass_binned_composite'] = bins
    except Exception:
        df_det['biomass_composite_bins'] = 0
        df_det['biomass_binned_composite'] = 0

    return df_det

