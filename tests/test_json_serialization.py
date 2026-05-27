"""
Test cases for JSON serialization of Sky objects
"""
import json
import tempfile
import pathlib
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord

import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from data3gc.sky_ndarray import Sky


def test_sky_to_dict():
    """Test conversion of Sky object to dictionary"""
    # Create a simple Sky object
    centrecoords = SkyCoord(ra=0*u.deg, dec=0*u.deg)
    npix = 512
    cellsize = 1.0 * u.arcsec
    freqs = [1.4 * u.GHz, 1.5 * u.GHz]
    
    sky = Sky(
        centrecoords=centrecoords,
        npix=npix,
        cellsize=cellsize,
        freqs=freqs,
        nfacets=0,
        stokes="I",
        skyname="TestSky"
    )
    
    # Add some dummy data
    sky.data["restored"] = np.random.randn(*sky.imshape).astype(np.float32)
    
    # Convert to dict
    sky_dict = sky.to_dict(include_data=True)
    
    # Verify structure
    assert sky_dict["name"] == "TestSky"
    assert sky_dict["npix"] == 512
    assert sky_dict["phasecenter"]["ra"] == 0.0
    assert sky_dict["phasecenter"]["dec"] == 0.0
    assert len(sky_dict["freqs"]) == 2
    assert "restored" in sky_dict["data"]
    print("✓ test_sky_to_dict passed")


def test_sky_to_json():
    """Test JSON string serialization"""
    centrecoords = SkyCoord(ra=10*u.deg, dec=45*u.deg)
    npix = 256
    cellsize = 0.5 * u.arcsec
    freqs = [1.4 * u.GHz]
    
    sky = Sky(
        centrecoords=centrecoords,
        npix=npix,
        cellsize=cellsize,
        freqs=freqs,
        nfacets=0,
        stokes="I",
        skyname="TestSky2"
    )
    
    # Convert to JSON
    json_str = sky.to_json(include_data=False, indent=2)
    
    # Verify it's valid JSON
    sky_dict = json.loads(json_str)
    assert sky_dict["name"] == "TestSky2"
    assert sky_dict["npix"] == 256
    print("✓ test_sky_to_json passed")


def test_sky_to_json_file():
    """Test writing Sky to JSON file"""
    centrecoords = SkyCoord(ra=15*u.deg, dec=60*u.deg)
    npix = 128
    cellsize = 1.0 * u.arcsec
    freqs = [1.4 * u.GHz, 1.5 * u.GHz, 1.6 * u.GHz]
    
    sky = Sky(
        centrecoords=centrecoords,
        npix=npix,
        cellsize=cellsize,
        freqs=freqs,
        nfacets=0,
        stokes="I",
        skyname="TestSky3"
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = pathlib.Path(tmpdir) / "test_sky.json"
        
        # Write to file
        sky.to_json_file(str(filepath), include_data=False)
        
        # Verify file exists
        assert filepath.exists()
        
        # Read and verify content
        with open(filepath, 'r') as f:
            loaded_dict = json.load(f)
        
        assert loaded_dict["name"] == "TestSky3"
        assert loaded_dict["npix"] == 128
        assert len(loaded_dict["freqs"]) == 3
        print("✓ test_sky_to_json_file passed")


def test_sky_from_dict():
    """Test creating Sky object from dictionary"""
    # Create and serialize
    centrecoords = SkyCoord(ra=20*u.deg, dec=30*u.deg)
    npix = 256
    cellsize = 0.5 * u.arcsec
    freqs = [1.4 * u.GHz, 1.5 * u.GHz]
    
    sky1 = Sky(
        centrecoords=centrecoords,
        npix=npix,
        cellsize=cellsize,
        freqs=freqs,
        nfacets=0,
        stokes="I",
        skyname="OriginalSky"
    )
    sky1.data["restored"] = np.random.randn(*sky1.imshape).astype(np.float32)
    
    # Convert to dict and back
    sky_dict = sky1.to_dict(include_data=True)
    sky2 = Sky.from_dict(sky_dict)
    
    # Verify properties
    assert sky2.name == sky1.name
    assert sky2.npix == sky1.npix
    assert sky2.phasecenter.ra.deg == sky1.phasecenter.ra.deg
    assert sky2.phasecenter.dec.deg == sky1.phasecenter.dec.deg
    assert len(sky2.freqs) == len(sky1.freqs)
    assert sky2.stokes == sky1.stokes
    assert "restored" in sky2.data
    assert sky2.data["restored"].shape == sky1.data["restored"].shape
    print("✓ test_sky_from_dict passed")


def test_sky_from_json():
    """Test creating Sky object from JSON string"""
    centrecoords = SkyCoord(ra=25*u.deg, dec=35*u.deg)
    npix = 512
    cellsize = 0.25 * u.arcsec
    freqs = [1.4 * u.GHz]
    
    sky1 = Sky(
        centrecoords=centrecoords,
        npix=npix,
        cellsize=cellsize,
        freqs=freqs,
        nfacets=0,
        stokes="I",
        skyname="JsonTestSky"
    )
    
    # Serialize to JSON and back
    json_str = sky1.to_json(include_data=False)
    sky2 = Sky.from_json(json_str)
    
    # Verify
    assert sky2.name == sky1.name
    assert sky2.npix == sky1.npix
    print("✓ test_sky_from_json passed")


def test_sky_from_json_file():
    """Test round-trip: file write and load"""
    centrecoords = SkyCoord(ra=30*u.deg, dec=40*u.deg)
    npix = 256
    cellsize = 1.0 * u.arcsec
    freqs = [1.4 * u.GHz, 1.5 * u.GHz]
    
    sky1 = Sky(
        centrecoords=centrecoords,
        npix=npix,
        cellsize=cellsize,
        freqs=freqs,
        nfacets=0,
        stokes="I",
        skyname="FilePersistenceSky"
    )
    sky1.data["restored"] = np.random.randn(*sky1.imshape).astype(np.float32)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = pathlib.Path(tmpdir) / "sky_roundtrip.json"
        
        # Write and load
        sky1.to_json_file(str(filepath), include_data=True)
        sky2 = Sky.from_json_file(str(filepath))
        
        # Verify all properties match
        assert sky2.name == sky1.name
        assert sky2.npix == sky1.npix
        assert sky2.nfacets == sky1.nfacets
        assert sky2.phasecenter.ra.deg == sky1.phasecenter.ra.deg
        assert sky2.phasecenter.dec.deg == sky1.phasecenter.dec.deg
        assert np.allclose(sky2.data["restored"], sky1.data["restored"])
        print("✓ test_sky_from_json_file passed")


def test_sky_with_facets_serialization():
    """Test serialization of Sky with facets"""
    centrecoords = SkyCoord(ra=0*u.deg, dec=0*u.deg)
    npix = 512
    cellsize = 1.0 * u.arcsec
    freqs = [1.4 * u.GHz]
    
    sky1 = Sky(
        centrecoords=centrecoords,
        npix=npix,
        cellsize=cellsize,
        freqs=freqs,
        nfacets=2,  # Create 4 facets (2x2)
        stokes="I",
        skyname="FacetedSky"
    )
    
    # Convert to dict
    sky_dict = sky1.to_dict(include_data=False)
    
    # Verify facets structure
    assert "facets" in sky_dict
    assert len(sky_dict["facets"]) == 4
    
    # Recreate from dict
    sky2 = Sky.from_dict(sky_dict)
    
    assert sky2.nfacets == sky1.nfacets
    assert len(sky2.facets) == len(sky1.facets)
    print("✓ test_sky_with_facets_serialization passed")


if __name__ == "__main__":
    print("Running JSON serialization tests...\n")
    
    test_sky_to_dict()
    test_sky_to_json()
    test_sky_to_json_file()
    test_sky_from_dict()
    test_sky_from_json()
    test_sky_from_json_file()
    test_sky_with_facets_serialization()
    
    print("\n✓ All tests passed!")
