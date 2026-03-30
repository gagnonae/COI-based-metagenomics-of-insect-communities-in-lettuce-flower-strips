import pandas as pd
import numpy as np
import re

# ----------------------------
# Robust taxonomy parsing
# ----------------------------

RANK_PREFIXES = {
    'k': 'kingdom',
    'p': 'phylum',
    'c': 'class',
    'o': 'order',
    'f': 'family',
    'g': 'genus',
    's': 'species'
}

RANK_PATTERN = re.compile(r'([kpcogfs])_{2,3}([^;,
\r]+)')
BAD_VALUES = {'', '_', '__', '___', 'unassigned', 'unknown', 'na', 'nan'}

def clean_tax_value(val):
    if val is None:
        return None
    val = val.strip()
    val = re.sub(r'_+$', '', val)
    val = re.sub(r'\s+', ' ', val)
    if val.lower() in BAD_VALUES:
        return None
    return val

def parse_taxonomy(tax_string):
    if not isinstance(tax_string, str):
        return {}
    if tax_string.lower().startswith('unassigned'):
        return {}
    if tax_string.startswith(('forward-', 'reverse-')):
        return {}
    ranks = {}
    for rank_code, value in RANK_PATTERN.findall(tax_string):
        rank = RANK_PREFIXES[rank_code]
        value = clean_tax_value(value)
        if value:
            ranks[rank] = value
    return ranks

def extract_order(tax_string):
    tax = parse_taxonomy(tax_string)
    order = tax.get('order')
    family = tax.get('family')
    if order == 'Hymenoptera' and family == 'Braconidae':
        return 'Braconidae'
    return order

def extract_genus_species(tax_string):
    tax = parse_taxonomy(tax_string)
    genus = tax.get('genus')
    species = tax.get('species')
    if species:
        species = re.sub(r'^(sp\.legit\?|aff\.legit\?|cf\.legit\?)\s*', '', species, flags=re.IGNORECASE)
        species = species.replace(' ', '_')
        species = clean_tax_value(species)
    if genus and species:
        return f"{genus}_{species}"
    if genus:
        return genus
    return None

# ----------------------------
# Parameters
# ----------------------------

OBSERVED_ABUNDANCE_FILE = 'Observed.matrix'
ANALYSIS_IDS = 'Analysis.ids'
CORRECTION_FACTORS = 'correction.factors'

SPIKE_IN_COLUMNS = [
    'Atta_mexicana',
    'Mecynorhina_torquata',
    'Bombyx_mandarina'
]

OUTPUT_FILE = 'corrected_proportions.tsv'

# ----------------------------
# Load data
# ----------------------------

print("\n[INFO] Loading observed abundance matrix...")
observed_df = pd.read_csv(OBSERVED_ABUNDANCE_FILE, sep=',', index_col=0)
print(f"[INFO] Raw observed_df shape: {observed_df.shape}")

observed_df = observed_df.iloc[:, :-2]
print(f"[INFO] After dropping last two columns: {observed_df.shape}")

analysis_ids = pd.read_csv(ANALYSIS_IDS, sep='\t', header=None)[0]
correction_factors_df = pd.read_csv(CORRECTION_FACTORS, sep='\t', index_col=0)
order_to_factor = correction_factors_df['factor'].to_dict()
print(f"[INFO] Loaded {len(order_to_factor)} correction factors")

# ----------------------------
# Filter samples
# ----------------------------

n_samples_before = observed_df.shape[0]
observed_df = observed_df.loc[observed_df.index.intersection(analysis_ids)]
print(f"[INFO] Filtered samples: {n_samples_before} → {observed_df.shape[0]}")

# ----------------------------
# Taxonomy parsing
# ----------------------------

print("\n[INFO] Parsing taxonomy strings...")
orders = {col: extract_order(col) for col in observed_df.columns}
new_columns = {col: extract_genus_species(col) or col for col in observed_df.columns}

missing_orders = [c for c, o in orders.items() if o is None]
if missing_orders:
    print(f"[WARN] {len(missing_orders)} taxa missing order assignment")
    print("       Example:", missing_orders[:3])

# ----------------------------
# Apply correction factors
# ----------------------------

print("\n[INFO] Applying correction factors...")
missing_factors = sorted({
    orders[col] for col in observed_df.columns
    if orders[col] not in order_to_factor and orders[col] is not None
})

if missing_factors:
    print(f"[WARN] Missing correction factors for {len(missing_factors)} orders")
    print("       Orders:", missing_factors)

factors = pd.Series(
    [order_to_factor.get(orders[col], 1.0) for col in observed_df.columns],
    index=observed_df.columns
)

corrected_df = observed_df.mul(factors, axis=1)
corrected_df = corrected_df.rename(columns=new_columns)

# ----------------------------
# Collapse duplicated columns AFTER renaming
# ----------------------------
n_cols_before = corrected_df.shape[1]
corrected_df = corrected_df.T.groupby(level=0).sum().T
print(f"[INFO] Collapsed duplicated columns: {n_cols_before} → {corrected_df.shape[1]}")

duplicates_after_rename = corrected_df.columns[corrected_df.columns.duplicated()]
if len(duplicates_after_rename) > 0:
    print("[WARN] Duplicates remain after collapsing:", duplicates_after_rename.tolist())

# ----------------------------
# Spike-in normalization
# ----------------------------

print("\n[INFO] Performing spike-in normalization...")
spike_in_cols = [c for c in SPIKE_IN_COLUMNS if c in corrected_df.columns]
print(f"[INFO] Spike-ins found: {spike_in_cols}")

if not spike_in_cols:
    raise ValueError("No spike-in columns found after renaming.")

spike_sums = corrected_df[spike_in_cols].sum(axis=1)
zero_spike_rows = spike_sums[spike_sums == 0].index.tolist()
if zero_spike_rows:
    print(f"[WARN] {len(zero_spike_rows)} samples have zero spike-in signal")
    print("       Samples:", zero_spike_rows[:3])

spike_sums = spike_sums.replace(0, np.nan)
normalized_df = corrected_df.div(spike_sums, axis=0) * 50000

# ----------------------------
# Proportions excluding spike-ins
# ----------------------------

print("\n[INFO] Normalizing non–spike-in proportions...")
non_spike_cols = normalized_df.columns.difference(spike_in_cols)
non_spike_sums = normalized_df[non_spike_cols].sum(axis=1).replace(0, np.nan)
normalized_df.loc[:, non_spike_cols] = normalized_df[non_spike_cols].div(non_spike_sums, axis=0)
print("[INFO] Example row sum (non-spike-ins):", normalized_df[non_spike_cols].iloc[0].sum())

# ----------------------------
# Add order row
# ----------------------------

order_row = pd.DataFrame(
    [{col: orders.get(orig_col, 'Unknown') for orig_col, col in new_columns.items()}],
    index=['Order']
)

final_df = pd.concat([order_row, normalized_df])
final_df.to_csv(OUTPUT_FILE, sep='\t')
print(f"\n[SUCCESS] Processed data saved to {OUTPUT_FILE}")
print(f"[INFO] Final table shape: {final_df.shape}")

# ----------------------------
# Sum proportions per order
# ----------------------------

print("\n[INFO] Calculating sum of proportions per order (excluding spike-ins)...")
non_spike_df = normalized_df.drop(columns=spike_in_cols, errors='ignore')

# Map renamed columns to orders
renamed_col_to_order = {new_columns[col]: orders[col] or 'Unknown' for col in new_columns}
col_to_order = {col: renamed_col_to_order.get(col, 'Unknown') for col in non_spike_df.columns}

order_df = non_spike_df.copy()
order_df.columns = [col_to_order[col] for col in non_spike_df.columns]
per_order_sum_df = order_df.groupby(axis=1, level=0).sum()

print("[INFO] Per-order sums (first 5 samples):")
print(per_order_sum_df.head())
per_order_sum_df.to_csv('per_order_proportions.tsv', sep='\t')
print("[INFO] Per-order proportions saved to 'per_order_proportions.tsv'")