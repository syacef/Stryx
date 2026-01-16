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
    RTSP_INPUT = "rtsp://127.0.0.1:8554/mystream"

    
    print("Creating Pipeline...")
    pipeline = Gst.Pipeline()

    # 1. Source - RTSP with explicit decoding chain
    source = Gst.ElementFactory.make("rtspsrc", "source")
    source.set_property("location", RTSP_INPUT)
    source.set_property("latency", 100)
    source.set_property("drop-on-latency", True)
    
    # Depayloader and parser for H264
    rtph264depay = Gst.ElementFactory.make("rtph264depay", "depay")
    h264parse = Gst.ElementFactory.make("h264parse", "parser")
    
    # NVIDIA H264 decoder
    nvv4l2decoder = Gst.ElementFactory.make("nvv4l2decoder", "decoder")

    # 2. Muxer (Required for DeepStream even with 1 source)
    streammux = Gst.ElementFactory.make("nvstreammux", "streammux")
    streammux.set_property("width", 320)
    streammux.set_property("height", 180)
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
    pipeline.add(rtph264depay)
    pipeline.add(h264parse)
    pipeline.add(nvv4l2decoder)
    pipeline.add(streammux)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(sink)

    print("Linking elements...")
    # Dynamic linking for rtspsrc (emits pad when stream is ready)
    def pad_added_handler(src, new_pad):
        print(f"Pad added: {new_pad.get_name()}")
        sink_pad = rtph264depay.get_static_pad("sink")
        if not sink_pad.is_linked():
            ret = new_pad.link(sink_pad)
            if ret == Gst.PadLinkReturn.OK:
                print("Successfully linked RTSP source to depayloader")
            else:
                print(f"Failed to link: {ret}")
    
    source.connect("pad-added", pad_added_handler)
    
    # Link the decode chain
    rtph264depay.link(h264parse)
    h264parse.link(nvv4l2decoder)
    
    # Link decoder to muxer (also needs dynamic pad)
    def decoder_pad_added(src, new_pad):
        sink_pad = streammux.get_request_pad("sink_0")
        if not new_pad.link(sink_pad) == Gst.PadLinkReturn.OK:
            print("Failed to link decoder to muxer")
        else:
            print("Successfully linked decoder to muxer")
    
    nvv4l2decoder.connect("pad-added", decoder_pad_added)
    
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
