import pandas as pd

def map_star_to_label(star):
    """
    Heuristically maps star ratings to sentiment classes:
    - 1 or 2 stars -> 0 (Negative)
    - 3 stars      -> 1 (Neutral)
    - 4 or 5 stars -> 2 (Positive)
    """
    try:
        star_val = int(star)
        if star_val in [1, 2]:
            return 0
        elif star_val == 3:
            return 1
        elif star_val in [4, 5]:
            return 2
    except (ValueError, TypeError):
        pass
    return 2  # Default to Positive if rating is missing or invalid

def correct_aspect_typos(df, aspect_column="LLM_Aspect"):
    """
    Corrects common aspect label typos in the DataFrame (e.g. 'Chất lương' -> 'Chất lượng').
    """
    if df is not None and aspect_column in df.columns:
        df[aspect_column] = df[aspect_column].astype(str).str.replace("Chất lương", "Chất lượng")
    return df

def load_classified_dataset(csv_path):
    """
    Helper function to load the LLM-classified reviews from CSV,
    correct spelling errors, and return a clean DataFrame.
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = correct_aspect_typos(df)
    return df
