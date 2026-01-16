import sys
import gi
import logging
from datetime import datetime
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import pyds

# Configure logging with DEBUG level for maximum verbosity
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('classification_log.txt'),
        logging.StreamHandler()
    ]
)

# Also enable GStreamer debug logging
import os
os.environ['GST_DEBUG'] = '2'  # 0=none, 1=error, 2=warning, 3=info, 4=debug, 5=trace

# Configuration
RTSP_INPUT = "rtsp://127.0.0.1:8554/mystream"
MODEL_ENGINE_PATH = "/app/model.engine"  # Path to TensorRT engine file
LABEL_FILE = "/app/labels.txt"  # Path to class labels file
NUM_CLASSES = 91
CONFIDENCE_THRESHOLD = 0.5

def load_labels(label_file):
    """Load class labels from file"""
    try:
        with open(label_file, 'r') as f:
            labels = [line.strip() for line in f.readlines()]
        return labels
    except:
        logging.warning(f"Could not load labels from {label_file}, using default class IDs")
        return [f"class_{i}" for i in range(NUM_CLASSES)]

# Load class labels
CLASS_LABELS = load_labels(LABEL_FILE)

def osd_sink_pad_buffer_probe(pad, info, u_data):
    """
    Probe function to intercept buffers and extract classification metadata
    """
    frame_number = 0
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    # Get metadata from buffer
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
        
        frame_number = frame_meta.frame_num
        l_obj = frame_meta.obj_meta_list
        
        # Track if we found any classifications in this frame
        classifications = []
        
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            
            # Get classifier metadata
            l_classifier = obj_meta.classifier_meta_list
            while l_classifier is not None:
                try:
                    classifier_meta = pyds.NvDsClassifierMeta.cast(l_classifier.data)
                    l_label = classifier_meta.label_info_list
                    
                    while l_label is not None:
                        try:
                            label_info = pyds.NvDsLabelInfo.cast(l_label.data)
                            class_id = label_info.result_class_id
                            confidence = label_info.result_prob
                            
                            if confidence >= CONFIDENCE_THRESHOLD:
                                class_name = CLASS_LABELS[class_id] if class_id < len(CLASS_LABELS) else f"class_{class_id}"
                                classifications.append({
                                    'class_id': class_id,
                                    'class_name': class_name,
                                    'confidence': confidence
                                })
                        except StopIteration:
                            break
                        try:
                            l_label = l_label.next
                        except StopIteration:
                            break
                except StopIteration:
                    break
                try:
                    l_classifier = l_classifier.next
                except StopIteration:
                    break
            
            try:
                l_obj = l_obj.next
            except StopIteration:
                break
        
        # Log non-empty classifications
        if classifications:
            log_message = f"Frame {frame_number}: "
            for cls in classifications:
                log_message += f"{cls['class_name']} ({cls['confidence']:.2f}), "
            logging.info(log_message.rstrip(', '))
        
        try:
            l_frame = l_frame.next
        except StopIteration:
            break
    
    return Gst.PadProbeReturn.OK

def bus_call(bus, message, loop, pipeline):
    t = message.type
    logging.debug(f"Bus message received: {t}")
    
    if t == Gst.MessageType.EOS:
        logging.info("End of stream received")
        sys.stdout.write("End of stream\n")
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        sys.stderr.write(f"Error: {err}: {debug}\n")
        logging.error(f"Pipeline error: {err}")
        logging.error(f"Debug info: {debug}")
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        sys.stderr.write(f"Warning: {err}: {debug}\n")
        logging.warning(f"Pipeline warning: {err}")
    elif t == Gst.MessageType.STATE_CHANGED:
        if message.src == pipeline:
            old_state, new_state, pending_state = message.parse_state_changed()
            logging.info(f"Pipeline state changed: {old_state.value_nick} → {new_state.value_nick}")
    elif t == Gst.MessageType.STREAM_STATUS:
        status_type, owner = message.parse_stream_status()
        logging.debug(f"Stream status: {status_type.value_nick} from {owner.get_name()}")
    elif t == Gst.MessageType.ELEMENT:
        struct = message.get_structure()
        if struct:
            logging.debug(f"Element message from {message.src.get_name()}: {struct.get_name()}")
    elif t == Gst.MessageType.ASYNC_DONE:
        logging.info("Pipeline async operations completed")
    elif t == Gst.MessageType.LATENCY:
        logging.debug("Latency message received, reconfiguring...")
    
    return True

def main():
    try:
        Gst.init(None)
        logging.info("✓ GStreamer initialized successfully")
    except Exception as e:
        logging.error(f"✗ Failed to initialize GStreamer: {e}")
        sys.exit(1)
    
    logging.info("=" * 60)
    logging.info("Starting DeepStream Classification Pipeline")
    logging.info(f"RTSP Input: {RTSP_INPUT}")
    logging.info(f"Model: {MODEL_ENGINE_PATH}")
    logging.info(f"Number of classes: {NUM_CLASSES}")
    logging.info("=" * 60)
    
    print("Creating Pipeline...")
    logging.debug("Creating pipeline...")
    
    try:
        pipeline = Gst.Pipeline()
        logging.info("✓ Pipeline created")
    except Exception as e:
        logging.error(f"✗ Failed to create pipeline: {e}")
        sys.exit(1)

    # 1. Source - RTSP with explicit decoding chain
    logging.debug("Creating RTSP source element...")
    try:
        source = Gst.ElementFactory.make("rtspsrc", "source")
        if not source:
            raise Exception("rtspsrc element creation returned None")
        logging.info("✓ RTSP source created")
        
        source.set_property("location", RTSP_INPUT)
        source.set_property("latency", 100)
        source.set_property("drop-on-latency", True)
        source.set_property("protocols", 0x00000007)
        source.set_property("retry", 5)
        source.set_property("timeout", 30000000)
        logging.debug(f"✓ RTSP source configured: {RTSP_INPUT}")
    except Exception as e:
        logging.error(f"✗ Failed to create/configure RTSP source: {e}")
        sys.exit(1)
    
    # Depayloader and parser for H264
    logging.debug("Creating H264 depayloader and parser...")
    try:
        rtph264depay = Gst.ElementFactory.make("rtph264depay", "depay")
        h264parse = Gst.ElementFactory.make("h264parse", "parser")
        if not rtph264depay or not h264parse:
            raise Exception("Failed to create depay or parser")
        logging.info("✓ H264 depayloader and parser created")
    except Exception as e:
        logging.error(f"✗ Failed to create H264 elements: {e}")
        sys.exit(1)
    
    # NVIDIA H264 decoder
    logging.debug("Creating H264 decoder...")
    try:
        decoder = Gst.ElementFactory.make("nvv4l2decoder", "decoder")
        if not decoder:
            logging.warning("nvv4l2decoder not available, falling back to avdec_h264")
            decoder = Gst.ElementFactory.make("avdec_h264", "decoder")
            if not decoder:
                raise Exception("Both nvv4l2decoder and avdec_h264 failed")
            logging.info("✓ Using CPU decoder (avdec_h264)")
        else:
            # Configure NVIDIA decoder for better stream handling
            decoder.set_property("skip-frames", 0)  # Don't skip any frames
            decoder.set_property("num-extra-surfaces", 1)
            logging.info("✓ Using NVIDIA decoder (nvv4l2decoder)")
    except Exception as e:
        logging.error(f"✗ Failed to create decoder: {e}")
        sys.exit(1)
    
    # Add videoconvert after decoder
    logging.debug("Creating videoconvert...")
    try:
        videoconvert = Gst.ElementFactory.make("videoconvert", "videoconvert")
        if not videoconvert:
            raise Exception("videoconvert creation failed")
        logging.info("✓ Videoconvert created")
    except Exception as e:
        logging.error(f"✗ Failed to create videoconvert: {e}")
        sys.exit(1)

    # 2. Muxer (Required for DeepStream)
    logging.debug("Creating nvstreammux...")
    try:
        streammux = Gst.ElementFactory.make("nvstreammux", "streammux")
        if not streammux:
            raise Exception("nvstreammux creation failed")
        streammux.set_property("width", 320)
        streammux.set_property("height", 180)
        streammux.set_property("batch-size", 1)
        streammux.set_property("batched-push-timeout", 4000000)
        streammux.set_property("live-source", True)
        logging.info("✓ Streammux created and configured")
    except Exception as e:
        logging.error(f"✗ Failed to create streammux: {e}")
        sys.exit(1)

    # 3. Inference - Primary Detector/Classifier
    logging.debug("Creating nvinfer...")
    try:
        pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
        if not pgie:
            raise Exception("nvinfer creation failed")
        pgie.set_property("config-file-path", "/app/config_infer_primary.txt")
        logging.info("✓ Inference engine created")
    except Exception as e:
        logging.error(f"✗ Failed to create inference engine: {e}")
        sys.exit(1)
    
    # 4. Converter & OSD
    logging.debug("Creating nvvideoconvert and nvdsosd...")
    try:
        nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
        nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
        if not nvvidconv or not nvosd:
            raise Exception("Failed to create converter or OSD")
        logging.info("✓ Video converter and OSD created")
    except Exception as e:
        logging.error(f"✗ Failed to create converter/OSD: {e}")
        sys.exit(1)

    # 5. Output - fakesink for headless operation
    logging.debug("Creating fakesink...")
    try:
        sink = Gst.ElementFactory.make("fakesink", "sink")
        if not sink:
            raise Exception("fakesink creation failed")
        sink.set_property("sync", False)
        logging.info("✓ Sink created")
    except Exception as e:
        logging.error(f"✗ Failed to create sink: {e}")
        sys.exit(1)
    
    logging.info("=" * 60)
    logging.info("All elements created successfully")
    logging.info("=" * 60)
    
    print("Adding elements to pipeline...")
    logging.debug("Adding elements to pipeline...")
    try:
        pipeline.add(source)
        pipeline.add(rtph264depay)
        pipeline.add(h264parse)
        pipeline.add(decoder)
        pipeline.add(videoconvert)
        pipeline.add(streammux)
        pipeline.add(pgie)
        pipeline.add(nvvidconv)
        pipeline.add(nvosd)
        pipeline.add(sink)
        logging.info("✓ All elements added to pipeline")
    except Exception as e:
        logging.error(f"✗ Failed to add elements to pipeline: {e}")
        sys.exit(1)

    print("Linking elements...")
    logging.info("=" * 60)
    logging.info("Linking pipeline elements...")
    logging.info("=" * 60)
    
    # Dynamic linking for rtspsrc
    def pad_added_handler(src, new_pad):
        try:
            pad_name = new_pad.get_name()
            logging.info(f"🔗 RTSP source pad added: {pad_name}")
            print(f"Pad added: {pad_name}")
            
            sink_pad = rtph264depay.get_static_pad("sink")
            if not sink_pad:
                logging.error("✗ Could not get depayloader sink pad")
                return
                
            if not sink_pad.is_linked():
                ret = new_pad.link(sink_pad)
                if ret == Gst.PadLinkReturn.OK:
                    logging.info("✓ Successfully linked RTSP source → depayloader")
                    print("Successfully linked RTSP source to depayloader")
                else:
                    logging.error(f"✗ Failed to link RTSP → depayloader: {ret}")
                    print(f"Failed to link: {ret}")
            else:
                logging.debug("Sink pad already linked, skipping")
        except Exception as e:
            logging.error(f"✗ Exception in pad_added_handler: {e}")
    
    source.connect("pad-added", pad_added_handler)
    logging.debug("✓ Connected pad-added signal for RTSP source")

    # Link the decode chain
    logging.debug("Linking decode chain...")
    
    # Add caps filter to ensure proper format for NVIDIA decoder
    logging.debug("Creating caps filter for decoder input...")
    try:
        decoder_caps = Gst.Caps.from_string("video/x-h264, stream-format=(string)byte-stream, alignment=(string)au")
        decoder_capsfilter = Gst.ElementFactory.make("capsfilter", "decoder_caps")
        decoder_capsfilter.set_property("caps", decoder_caps)
        pipeline.add(decoder_capsfilter)
        logging.info("✓ Decoder caps filter created")
    except Exception as e:
        logging.error(f"✗ Failed to create decoder caps filter: {e}")
        sys.exit(1)
    
    try:
        if not rtph264depay.link(h264parse):
            raise Exception("Failed to link depayloader → parser")
        logging.info("✓ Linked depayloader → parser")
        
        if not h264parse.link(decoder_capsfilter):
            raise Exception("Failed to link parser → caps filter")
        logging.info("✓ Linked parser → caps filter")
        
        if not decoder_capsfilter.link(decoder):
            raise Exception("Failed to link caps filter → decoder")
        logging.info("✓ Linked caps filter → decoder")
    except Exception as e:
        logging.error(f"✗ Failed to link decode chain: {e}")
        sys.exit(1)
    
    # Link decoder output (handle both sync and async pad creation)
    def decoder_pad_added(src, new_pad):
        try:
            pad_name = new_pad.get_name()
            logging.info(f"🔗 Decoder pad added: {pad_name}")
            logging.info("🎉 DECODER IS NOW PRODUCING OUTPUT!")
            print(f"Decoder pad added: {pad_name}")
            
            # First link to videoconvert
            convert_sink = videoconvert.get_static_pad("sink")
            if not convert_sink:
                logging.error("✗ Could not get videoconvert sink pad")
                return
                
            if not convert_sink.is_linked():
                ret = new_pad.link(convert_sink)
                if ret == Gst.PadLinkReturn.OK:
                    logging.info("✓ Linked decoder → videoconvert")
                    print("✓ Linked decoder to videoconvert")
                else:
                    logging.error(f"✗ Failed to link decoder → videoconvert: {ret}")
                    print(f"✗ Failed to link decoder to videoconvert: {ret}")
            else:
                logging.debug("Videoconvert sink already linked")
        except Exception as e:
            logging.error(f"✗ Exception in decoder_pad_added: {e}")
    
    decoder.connect("pad-added", decoder_pad_added)
    logging.debug("✓ Connected pad-added signal for decoder")
    
    # Add probes to see data flow
    probe_count = {'parse': 0, 'decoder': 0}
    
    def probe_h264parse(pad, info, user_data):
        if probe_count['parse'] == 0:
            logging.info("📊 Data flowing through h264parse!")
            probe_count['parse'] += 1
        return Gst.PadProbeReturn.OK
    
    def probe_decoder_sink(pad, info, user_data):
        if probe_count['decoder'] == 0:
            logging.info("📊 Data arriving at decoder input!")
            probe_count['decoder'] += 1
        return Gst.PadProbeReturn.OK
    
    # Add probes (only log first buffer)
    h264parse_src = h264parse.get_static_pad("src")
    if h264parse_src:
        h264parse_src.add_probe(Gst.PadProbeType.BUFFER, probe_h264parse, 0)
        logging.debug("✓ Added probe to h264parse output")
    
    decoder_sink = decoder.get_static_pad("sink")
    if decoder_sink:
        decoder_sink.add_probe(Gst.PadProbeType.BUFFER, probe_decoder_sink, 0)
        logging.debug("✓ Added probe to decoder input")
    
    # Try static link videoconvert to muxer
    logging.debug("Attempting to link videoconvert → muxer...")
    try:
        mux_sink_pad = streammux.get_request_pad("sink_0")
        if not mux_sink_pad:
            raise Exception("Could not get muxer sink pad")
        logging.debug(f"✓ Got muxer sink pad: {mux_sink_pad.get_name()}")
        
        convert_src_pad = videoconvert.get_static_pad("src")
        if not convert_src_pad:
            raise Exception("Could not get videoconvert src pad")
        logging.debug("✓ Got videoconvert src pad")
        
        ret = convert_src_pad.link(mux_sink_pad)
        if ret == Gst.PadLinkReturn.OK:
            logging.info("✓ Successfully linked videoconvert → muxer")
            print("✓ Static link videoconvert → muxer successful")
        else:
            logging.warning(f"Static link failed: {ret}, pipeline may use dynamic linking")
            print(f"Static link returned: {ret}")
    except Exception as e:
        logging.warning(f"Static linking failed: {e}, will try dynamic linking")
        print(f"Using dynamic linking: {e}")
    
    # Link the rest of the pipeline
    logging.debug("Linking post-muxer pipeline...")
    try:
        if not streammux.link(pgie):
            raise Exception("Failed to link muxer → inference")
        logging.info("✓ Linked muxer → inference")
        
        if not pgie.link(nvvidconv):
            raise Exception("Failed to link inference → converter")
        logging.info("✓ Linked inference → converter")
        
        if not nvvidconv.link(nvosd):
            raise Exception("Failed to link converter → OSD")
        logging.info("✓ Linked converter → OSD")
        
        if not nvosd.link(sink):
            raise Exception("Failed to link OSD → sink")
        logging.info("✓ Linked OSD → sink")
        
        logging.info("=" * 60)
        logging.info("✓ Pipeline fully linked!")
        logging.info("=" * 60)
    except Exception as e:
        logging.error(f"✗ Failed to link pipeline: {e}")
        sys.exit(1)

    # Add probe to OSD sink pad to intercept metadata
    logging.debug("Adding metadata probe...")
    try:
        osdsinkpad = nvosd.get_static_pad("sink")
        if not osdsinkpad:
            logging.warning("Could not get OSD sink pad")
            sys.stderr.write("Unable to get sink pad of nvosd\n")
        else:
            osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)
            logging.info("✓ Metadata probe added")
    except Exception as e:
        logging.error(f"✗ Failed to add probe: {e}")

    # Start
    logging.info("=" * 60)
    logging.info("Setting up main loop and bus...")
    try:
        loop = GLib.MainLoop()
        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", bus_call, loop, pipeline)
        logging.info("✓ Bus connected")
    except Exception as e:
        logging.error(f"✗ Failed to setup bus: {e}")
        sys.exit(1)

    print("Starting pipeline...")
    logging.info("=" * 60)
    logging.info("🚀 Setting pipeline to PLAYING state...")
    logging.info("=" * 60)
    
    try:
        ret = pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            logging.error("✗ Unable to set pipeline to PLAYING")
            sys.exit(1)
        elif ret == Gst.StateChangeReturn.NO_PREROLL:
            logging.info("Pipeline is live (no preroll needed)")
        elif ret == Gst.StateChangeReturn.ASYNC:
            logging.info("Pipeline changing state asynchronously...")
        elif ret == Gst.StateChangeReturn.SUCCESS:
            logging.info("✓ Pipeline PLAYING")
    except Exception as e:
        logging.error(f"✗ Exception setting state: {e}")
        sys.exit(1)
    
    logging.info("Waiting for stream... (may take a few seconds)")
    
    try:
        loop.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        logging.info("Pipeline stopped by user")
    except Exception as e:
        print(f"Exception: {e}")
        logging.error(f"Pipeline exception: {e}")
        import traceback
        traceback.print_exc()

    pipeline.set_state(Gst.State.NULL)
    logging.info("Pipeline stopped")

if __name__ == '__main__':
    main()
