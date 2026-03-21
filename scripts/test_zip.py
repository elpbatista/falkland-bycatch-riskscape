import zipfile

z = zipfile.ZipFile(
    "data/raw/wind/derived-era5-single-levels-daily-statistics_2014_01.zip"
)

print(z.namelist())