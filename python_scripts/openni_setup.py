from openni import openni2

def initialize_kinect():
    openni2.initialize()
    dev = openni2.Device.open_any()
    depth_stream = dev.create_depth_stream()
    color_stream = dev.create_color_stream()
    depth_stream.start()
    color_stream.start()
    dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)
    return dev, depth_stream, color_stream

def shutdown_kinect():
    openni2.unload()
