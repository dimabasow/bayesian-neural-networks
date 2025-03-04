import polars as pl


def drop_constant_columns(df: pl.DataFrame) -> pl.DataFrame:
    columns_to_drop = []
    for column in df.columns:
        series = df[column]
        series_drop_null = series.drop_nulls().drop_nans()
        if series_drop_null.len() == 0:
            columns_to_drop.append(column)
        elif (
            series_drop_null.len() == series.len()
            and (series_drop_null[0] == series_drop_null).all()
        ):
            columns_to_drop.append(column)
    df = df.drop(*columns_to_drop)
    return df
