
from __future__ import annotations

import re
import numpy as np
import pandas as pd


def build_cleaned_df(raw_csv_path) -> pd.DataFrame:
    df_local = pd.read_csv(raw_csv_path)

    df_cleaned = df_local.copy()

    mpg_split = df_cleaned["mpg"].astype(str).str.split("-", expand=True)
    df_cleaned["mpg_avg"] = (
        mpg_split
        .apply(pd.to_numeric, errors="coerce")
        .mean(axis=1)
    )
    df_cleaned = df_cleaned.drop(columns="mpg")

    df_cleaned = df_cleaned.dropna().reset_index(drop=True)

    bin_cols = ["accidents_or_damage", "one_owner", "personal_use_only"]
    for c in bin_cols:
        if c in df_cleaned.columns:
            df_cleaned[c] = df_cleaned[c].round().astype("Int64")

    return df_cleaned


def build_features_df(df_cleaned: pd.DataFrame) -> pd.DataFrame:
    # start from cleaned df already in memory
    y = np.log1p(df_cleaned["price"])
    X = df_cleaned.drop(columns=["price", "seller_name"], errors="ignore").copy()

    REFERENCE_YEAR = 2023 #The dataset ends at the beginning of 2023

    # age and mileage per year
    X["age"] = REFERENCE_YEAR - X["year"]
    X["age_for_mpy"] = X["age"].clip(lower=1)
    X["mileage_per_year"] = X["mileage"] / X["age_for_mpy"]
    X = X.drop(columns=["age_for_mpy", "year"])

    def normalize_brand(s: str) -> str:
        return (
            str(s)
            .strip()
            .upper()
            .replace("-", " ")
        )

    LUXURY_BRANDS = {
        "ACURA", "ALFA ROMEO", "ASTON MARTIN", "AUDI", "BENTLEY", "BMW", "BUGATTI", "CADILLAC", "FERRARI", "GENESIS", "INFINITI", "JAGUAR", "LAMBORGHINI", "LAND ROVER", "LEXUS", "LINCOLN", "LOTUS", "MASERATI", "MCLAREN", "MERCEDES BENZ", "POLESTAR", "PORSCHE", "ROLLS-ROYCE", "TESLA", "VOLVO"
    }

    X["manufacturer_norm"] = X["manufacturer"].apply(normalize_brand)

    X["is_luxury_brand"] = (
        X["manufacturer_norm"].isin(LUXURY_BRANDS).astype("int64")
    )

    X["luxury_age_interaction"] = X["is_luxury_brand"] * X["age"]

    # Drop helper column
    X = X.drop(columns=["manufacturer_norm"])

    def parse_engine(engine):
        s = str(engine).strip().upper()

        m_l = re.search(r"(\d(?:\.\d)?)\s*L\b", s)
        liters = float(m_l.group(1)) if m_l else np.nan

        m_c = re.search(r"\b([IVH])\s*[- ]?\s*(\d+)\b", s)
        if m_c:
            layout = m_c.group(1)
            cylinders = int(m_c.group(2))
        else:
            layout = np.nan
            cylinders = np.nan

        turbo = int(("TURBO" in s) or ("TWIN TURBO" in s) or ("SUPERCHARG" in s))

        hybrid = int(
            ("HYBRID" in s) or
            ("GAS/ELECTRIC" in s) or
            ("PHEV" in s) or
            ("PLUG-IN" in s) or
            ("ELECTRIC" in s and "GAS" in s)
        )

        return pd.Series({
            "engine_liters": liters,
            "engine_cylinders": cylinders,
            "engine_layout": layout,
            "engine_turbo": turbo,
            "engine_hybrid": hybrid
        })

    engine_features = X["engine"].apply(parse_engine)
    X = pd.concat([X.drop(columns=["engine"]), engine_features], axis=1)
    X["engine_layout"] = X["engine_layout"].fillna("unknown")

    def normalize_transmission(t):
        if pd.isna(t):
            return "unknown"
        t = str(t).lower().strip()
        t = t.replace("a/t", "automatic")
        t = t.replace("auto", "automatic")
        return t

    def transmission_type(t):
        t = normalize_transmission(t)
        if t == "unknown" or "not specified" in t:
            return "unknown"
        if "manual" in t:
            return "manual"
        if ("cvt" in t) or ("variable" in t) or ("ivt" in t) or ("ecvt" in t):
            return "cvt"
        return "automatic"

    def transmission_gears(t):
        t = normalize_transmission(t)
        match = re.search(r"(\d+)[-\s]?speed", t)
        return int(match.group(1)) if match else np.nan

    X["transmission_clean"] = X["transmission"].apply(transmission_type)
    X["transmission_gears"] = X["transmission"].apply(transmission_gears)
    X = X.drop(columns=["transmission"])
    X["transmission_gears_missing"] = X["transmission_gears"].isna().astype(int)

    def base_color(c):
        c = str(c).lower().strip()
        if "black" in c or "ebony" in c:
            return "black"
        if "blue" in c or "deep cerulean" in c:
            return "blue"
        if "red" in c or "scarlet ember" in c or "maroon" in c or "dark cherry" in c:
            return "red"
        if "green" in c or "army green" in c or "f8 green" in c or "dark moss" in c or "sarge green clearcoat" in c:
            return "green"
        if "white" in c or "pearl" in c or "whiite" in c:
            return "white"
        if "gray" in c or "silver" in c or "grey" in c or "gun" in c or "steel" in c or "magnetic" in c or " metal " in c or "carbon" in c or "granite" in c or "graphite" in c:
            return "gray"
        return "other"

    X["exterior_color_base"] = X["exterior_color"].apply(base_color)
    X = X.drop(columns=["exterior_color"])

    def interior_color_base(c):
        c = str(c).lower().strip()
        if "black" in c or "ebony" in c or "jet" in c:
            return "black"
        if "red" in c:
            return "red"
        if "gray" in c or "grey" in c or "graphite" in c or "charcoal" in c or "shale" in c or "steel" in c or "pewter" in c or "slate" in c:
            return "gray"
        if "brown" in c or "cappuccino" in c or "mocha" in c or "espresso" in c or "cocoa" in c or "coffee" in c or "nutmeg" in c or "walnut" in c or "chestnut" in c or "hazelnut" in c or "roast" in c:
            return "brown"
        if "beige" in c or "tan" in c or "taupe" in c or "sand" in c or "ash" in c or "camel" in c or "cognac" in c or "parchment" in c or "stone" in c or "wheat" in c or "sandstone" in c or "cement" in c or "almond" in c or "blond" in c or "neutral" in c:
            return "beige"
        if "white" in c or "ivory" in c or "cream" in c:
            return "white"
        return "other"

    X["interior_color_base"] = X["interior_color"].apply(interior_color_base)
    X = X.drop(columns=["interior_color"])

    # casts
    numeric_casts = {
        "age": "int64",
        "mileage_per_year": "float64",
        "engine_liters": "float64",
        "engine_cylinders": "float64",
        "transmission_gears": "float64",
        "engine_turbo": "int64",
        "engine_hybrid": "int64",
        "transmission_gears_missing": "int64",
    }
    for col, dtype in numeric_casts.items():
        if col in X.columns:
            X[col] = X[col].astype(dtype)

    for col in ["engine_layout", "transmission_clean", "exterior_color_base", "interior_color_base"]:
        if col in X.columns:
            X[col] = X[col].astype("object")

    # Save X and y 
    features_df = X.copy()
    features_df["target_log1p_price"] = y.values
    return features_df
