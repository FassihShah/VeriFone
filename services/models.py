from pydantic import BaseModel
from typing import Optional


class UsedMobile(BaseModel):
    brand: Optional[str] = None
    model: Optional[str] = None
    ram: Optional[str] = None
    storage: Optional[str] = None
    condition: Optional[int] = None
    pta_approved: Optional[bool] = None        

    is_panel_changed: Optional[bool] = None
    screen_crack: Optional[bool] = None
    panel_dot: Optional[bool] = None
    panel_line: Optional[bool] = None
    panel_shade: Optional[bool] = None
    camera_lens_ok: Optional[bool] = None
    fingerprint_ok: Optional[bool] = None

    with_box: Optional[bool] = None
    with_charger: Optional[bool] = None

    price: Optional[int] = None
    city: Optional[str] = None

    listing_source: Optional[str] = None     
    images: Optional[list[str]] = None
    post_date: Optional[str] = None


class NewMobile(BaseModel):
    brand: Optional[str] = None
    model: Optional[str] = None

    ram: Optional[str] = None
    storage: Optional[str] = None
    os: Optional[str] = None
    release_year: Optional[int] = None

    screen_size: Optional[str] = None
    screen_resolution: Optional[str] = None

    battery_capacity: Optional[str] = None

    main_camera: Optional[str] = None
    selfie_camera: Optional[str] = None

    chipset: Optional[str] = None
    cpu: Optional[str] = None
    gpu: Optional[str] = None

    network: Optional[str] = None
    network_bands: Optional[str] = None

    sim: Optional[str] = None

    weight: Optional[str] = None
    dimensions: Optional[str] = None

    usb: Optional[str] = None
    sensors: Optional[str] = None

    price: Optional[str] = None    
