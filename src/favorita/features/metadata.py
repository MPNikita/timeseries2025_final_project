from __future__ import annotations

import pandas as pd


def _encode_metadata_codes(items: pd.DataFrame, stores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Encode static item/store metadata into integer categorical codes."""
    item_meta = items[["item_nbr", "class", "perishable"]].copy()
    item_meta["family_code"] = pd.Categorical(items["family"]).codes.astype("int16")
    item_meta["class"] = item_meta["class"].astype("int32")
    item_meta["perishable"] = item_meta["perishable"].astype("int8")

    store_meta = stores[["store_nbr", "cluster"]].copy()
    store_meta["city_code"] = pd.Categorical(stores["city"]).codes.astype("int16")
    store_meta["state_code"] = pd.Categorical(stores["state"]).codes.astype("int16")
    store_meta["type_code"] = pd.Categorical(stores["type"]).codes.astype("int16")
    store_meta["cluster"] = store_meta["cluster"].astype("int16")
    return item_meta, store_meta

