"""Native capture with a Revopoint MetroY / MetroY Ultra, without Revo Metro.

    hid        the scanner's HID command channel: read files off it, write acquisition registers
    v4l2       continuous capture of the stereo IR stream (Linux)
    laserfile  the factory laser line calibration (metroExtra.bin)
    stripes    one laser frame -> 3D points in millimetres, through the factory stereo calibration
    scanner    ties them together: connect, read the calibration off the device, stream, triangulate

The protocol and every number here are reverse-engineered; docs/metroy-protocol.md is the reference and
docs/metroy-handoff.md the state of play. Nothing in this package is imported by the rest of CloudClean unless the
MetroY driver is used, so OpenCV stays an optional dependency (``pip install -e .[metroy]``).
"""
