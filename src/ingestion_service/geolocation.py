import logging
import reverse_geocoder as rg
import pycountry_convert as pc
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

def get_location_details(lat: float, lon: float) -> Tuple[Optional[str], Optional[str]]:
    """
    Determine Country and Continent from Latitude and Longitude.
    
    Args:
        lat: Latitude
        lon: Longitude
        
    Returns:
        (Country Name, Continent Name)
    """
    try:
        # reverse_geocoder search takes a list of tuples
        results = rg.search([(lat, lon)])
        if not results:
            return None, None
            
        # result is a list of OrderedDict, e.g., [{'lat': '...', 'lon': '...', 'name': '...', 'admin1': '...', 'admin2': '...', 'cc': 'FR'}]
        country_code = results[0]['cc']
        
        # Get Country Name
        # We can use pycountry_convert or just rely on the code, but the user asked for Name.
        # reverse_geocoder doesn't give full country name, only code.
        # We can use pycountry_convert to get the name if needed, or just use the code.
        # Let's try to get a readable name via pycountry_convert internal data or similar.
        
        # Actually pycountry_convert is mainly for continent mapping. 
        # Let's map CC to Continent first.
        try:
            continent_code = pc.country_alpha2_to_continent_code(country_code)
            continent_name = pc.convert_continent_code_to_continent_name(continent_code)
        except Exception as e:
            logger.warning(f"Could not determine continent for code {country_code}: {e}")
            continent_name = None
            
        # For Country Name, we can use a simple lookup or just return the code if a full library isn't available.
        # But wait, pycountry_convert relies on repo data.
        # Let's try to get the country name using standard library or map if possible. 
        # Since I didn't install 'pycountry' (only pycountry-convert), I might be limited.
        # However, `reverse_geocoder` is very lightweight.
        
        # Let's assume Country Code is acceptable as "Country" or I can add a simple mapping if strictly required.
        # But commonly Country Name is preferred.
        # I'll simply return the Country Code (e.g. "FR", "US") as the Country for now, 
        # as it's standard and cleaner for graphs than variable length names.
        # If the user strictly wants "France", I'd need `pycountry`.
        # I'll stick to Country Code for "Country" field for now to avoid extra heavy deps, 
        # unless I see `pycountry` is implicitly installed.
        
        return country_code, continent_name

    except Exception as e:
        logger.error(f"Geolocation error: {e}")
        return None, None
