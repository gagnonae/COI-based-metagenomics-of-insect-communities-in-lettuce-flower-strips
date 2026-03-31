# STEP 1: Import required libraries
import pandas as pd
import numpy as np
from scipy.stats import linregress

# STEP 2: Helper functions
def calculate_r_squared_and_p_value(observed, theoretical):
    slope, intercept, r_value, p_value, std_err = linregress(theoretical, observed)
    return r_value ** 2, p_value

def gmean_pandas(series):
    series = series[(series > 0) & np.isfinite(series)]
    return np.exp(np.log(series).mean()) if len(series) > 0 else np.nan

# STEP 3: Parameters
TRUE_ABUNDANCE_FILE = 'Theoretical.matrix'
OBSERVED_ABUNDANCE_FILE = 'ObservedEs.matrix'
SPECIES_KEY_FILE = 'Species_key_order.fofn'

SPIKE_IN_COLUMNS = ['Atta_mexicana', 'Mecynorhina_torquata', 'Bombyx_mandarina']
TREATMENTS = ['E1', 'E3', 'E4', 'E5', 'E6']

# STEP 4: Load data
observed_df = pd.read_csv(OBSERVED_ABUNDANCE_FILE, sep='\t', index_col=0)
true_df = pd.read_csv(TRUE_ABUNDANCE_FILE, sep='\t', index_col=0)

# Collapse duplicated columns
observed_df = observed_df.groupby(observed_df.columns, axis=1).sum()
observed_df_spike = observed_df.groupby(observed_df.columns, axis=1).sum()
true_df = true_df.groupby(true_df.columns, axis=1).sum()

# Match columns
common_cols = observed_df.columns.intersection(true_df.columns)
observed_df = observed_df[common_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
true_df = true_df[common_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

# STEP 5: Spike-in normalization
spike_sums = observed_df_spike[SPIKE_IN_COLUMNS].sum(axis=1)
normalized_obs = observed_df.div(spike_sums, axis=0) * 50000
normalized_obs = normalized_obs.div(normalized_obs.sum(axis=1), axis=0)

# STEP 6: Load species order key
order_key = pd.read_csv(SPECIES_KEY_FILE, sep='\t').set_index('Species')

# STEP 7: Build ONE clean species x treatment table
records = []

for treatment in TREATMENTS:
    obs = normalized_obs[normalized_obs.index.str.startswith(treatment)].sum()
    theo = true_df[true_df.index.str.startswith(treatment)].sum()
    raw = observed_df[observed_df.index.str.startswith(treatment)].sum()

    df = (
        pd.DataFrame({
            'Observed_Abundance': obs,
            'Theoretical_Abundance': theo,
            'Raw_Observed_Abundance': raw
        })
        .reset_index()
        .rename(columns={'index': 'Species'})
        .merge(order_key, left_on='Species', right_index=True)
    )

    df['Treatment'] = treatment
    records.append(df)

full_df = pd.concat(records, ignore_index=True)

# STEP 8: Keep rows with non-zero observed OR theoretical
nonzero_df = full_df[
    (full_df['Observed_Abundance'] > 0) |
    (full_df['Theoretical_Abundance'] > 0)
].copy()

# STEP 9: Compute correction factors ONLY when both > 0
valid_cf_df = nonzero_df[
    (nonzero_df['Observed_Abundance'] > 0) &
    (nonzero_df['Theoretical_Abundance'] > 0)
].copy()

valid_cf_df['Correction_Factor'] = (
    valid_cf_df['Theoretical_Abundance'] /
    valid_cf_df['Observed_Abundance']
)

# STEP 10: Order-level correction factors (per treatment)
order_cf_treatment = (
    valid_cf_df
    .groupby(['Treatment', 'Order'])['Correction_Factor']
    .apply(gmean_pandas)
    .reset_index()
)

nonzero_df = nonzero_df.merge(
    order_cf_treatment,
    on=['Treatment', 'Order'],
    how='left'
)

# STEP 11: Apply treatment-specific correction
nonzero_df['Corrected_Observed_Abundance'] = (
    nonzero_df['Observed_Abundance'] *
    nonzero_df['Correction_Factor']
)

# Normalize within treatment
nonzero_df['Corrected_Observed_Abundance'] = (
    nonzero_df
    .groupby('Treatment')['Corrected_Observed_Abundance']
    .transform(lambda x: x / x.sum() if x.sum() > 0 else x)
)

# STEP 12: Global correction factors
global_cf = (
    valid_cf_df
    .groupby('Order')['Correction_Factor']
    .apply(gmean_pandas)
    .reset_index()
    .rename(columns={'Correction_Factor': 'Correction_Factor_Global'})
)

nonzero_df = nonzero_df.merge(global_cf, on='Order', how='left')

# STEP 13: Apply global correction
nonzero_df['Global_Corrected_Observed_Abundance'] = (
    nonzero_df['Observed_Abundance'] *
    nonzero_df['Correction_Factor_Global']
)

nonzero_df['Global_Corrected_Observed_Abundance'] = (
    nonzero_df
    .groupby('Treatment')['Global_Corrected_Observed_Abundance']
    .transform(lambda x: x / x.sum() if x.sum() > 0 else x)
)

# STEP 14: Final sanity check
assert not nonzero_df.duplicated(
    subset=['Species', 'Treatment']
).any(), "Duplicate species x treatment rows detected!"

# STEP 15: Output
nonzero_df.to_csv('processed_no_E2_file.csv', index=False)

# Compute number of observations per order for global correction factors
obs_per_order = valid_cf_df.groupby('Order').size().reset_index(name='Num_Observations')

# Merge with global_cf to get summary
global_summary = global_cf.merge(obs_per_order, on='Order')

print("Processing complete")
print("Species retained if observed OR theoretical > 0")
print("Correction factors computed only when both > 0")
print("One row per Species x Treatment")
print("Global correction factors per order:")
print(global_summary.to_string(index=False))