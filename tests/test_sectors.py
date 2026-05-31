from quant.data.sectors import load_sectors


def test_load_sectors_maps_instrument_to_sector(fake_lake):
    root, _, _ = fake_lake
    sec = load_sectors(market="us", root=root)
    assert sec["AAA"] == "Tech"
    assert sec["BBB"] == "Tech"
    assert sec["CCC"] == "Energy"
