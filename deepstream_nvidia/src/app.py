import sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

def bus_call(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        sys.stdout.write("End of stream\n")
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        sys.stderr.write(f"Error: {err}: {debug}\n")
        loop.quit()
    return True

def main():
    Gst.init(None)

    # Standard DeepStream Pipeline
    # source (RTSP) -> h264depay -> h264parse -> nvv4l2decoder -> nvstreammux -> nvvideoconvert -> nvdsosd -> nvvideoconvert -> nvv4l2h264enc -> rtph264pay -> udpsink
    
    # NOTE: Replace this URI with your camera or a sample public RTSP stream
    RTSP_INPUT = "file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4"

    
    print("Creating Pipeline...")
    pipeline = Gst.Pipeline()

    # 1. Source (Simplification: using uridecodebin for automatic handling)
    source = Gst.ElementFactory.make("uridecodebin", "source")
    source.set_property("uri", RTSP_INPUT)

    # 2. Muxer (Required for DeepStream even with 1 source)
    streammux = Gst.ElementFactory.make("nvstreammux", "streammux")
    streammux.set_property("width", 1920)
    streammux.set_property("height", 1080)
    streammux.set_property("batch-size", 1)
    streammux.set_property("batched-push-timeout", 4000000)

    # 3. Inference (Fake sink for baseline - usually nvinfer goes here)
    # We will skip nvinfer for a pure connectivity test
    
    # 4. Converter & OSD
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")

    # 5. Output (RTSP Server Sink setup usually requires a full server code)
    # For this baseline, we will use fakesink to prove the pipeline runs, 
    # OR nveglglessink if you have X11 forwarding.
    # Let's use fakesink to ensure it runs headless without crashing.
    sink = Gst.ElementFactory.make("fakesink", "sink")
    
    print("Adding elements to pipeline...")
    pipeline.add(source)
    pipeline.add(streammux)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(sink)

    print("Linking elements...")
    # Dynamic linking for uridecodebin
    def pad_added_handler(src, new_pad):
        sink_pad = streammux.get_request_pad("sink_0")
        if not new_pad.link(sink_pad) == Gst.PadLinkReturn.OK:
            print("Failed to link decoder to muxer")
    
    source.connect("pad-added", pad_added_handler)
    
    # Link the rest
    streammux.link(nvvidconv)
    nvvidconv.link(nvosd)
    nvosd.link(sink)

    # Start
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    print("Starting pipeline...")
    pipeline.set_state(Gst.State.PLAYING)
    
    try:
        loop.run()
    except:
        pass

    pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    main()
