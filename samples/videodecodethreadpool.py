import pyRocVideoDecode.decoder as dec
import pyRocVideoDecode.demuxer as dmx
import datetime
import sys
import argparse
import os
from hip import hip

def HipCheck(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    return result


def DecProcEmpty():
    print("info: in DecProcEmpty")
    return

def DecProc(input_file_path, device_id):
    print("info: in DecProc with args")
    p_frames = 0
    p_fps = 0.0
    # demuxer instance
    demuxer = dmx.demuxer(input_file_path)

    # get the used coded id
    codec_id = dec.GetRocDecCodecID(demuxer.GetCodecId())

    # decoder instance
    viddec = dec.decoder(
        device_id = device_id,
        mem_type = 1,
        codec = codec_id,
        b_force_zero_latency = False,
        crop_rect = None,
        max_width = 0,
        max_height = 0,
        clk_rate = 1000)

    # Get GPU device information
    cfg = viddec.GetGpuInfo()

    #  print some GPU info out
    print("\ninfo: Input file: " +
          input_file_path +
          '\n' +
          "info: Using GPU device " +
          str(device_id) +
          " - " +
          cfg.device_name +
          "[" +
          cfg.gcn_arch_name +
          "] on PCI bus " +
          str(cfg.pci_bus_id) +
          ":" +
          str(cfg.pci_domain_id) +
          "." +
          str(cfg.pci_device_id))
    n_frame = 0
    total_dec_time = 0.0

    while True:
        start_time = datetime.datetime.now()
        packet = demuxer.DemuxFrame()

        n_frame_returned = viddec.DecodeFrame(packet)

        # measure after completing a whole frame
        end_time = datetime.datetime.now()
        time_per_frame = end_time - start_time
        total_dec_time = total_dec_time + time_per_frame.total_seconds()

        # increament frames counter
        n_frame += n_frame_returned

        if (packet.frame_size <= 0):  # EOF: no more to decode
            break

    # beyond the decoding loop
    n_frame += viddec.GetNumOfFlushedFrames()

    p_frames = n_frame

    if (n_frame > 0 and total_dec_time > 0):
        time_per_frame = (total_dec_time / n_frame) * 1000
        session_overhead = viddec.GetDecoderSessionOverHead(os.getpid())
        if (session_overhead == None):
            session_overhead = 0
        time_per_frame -= (session_overhead / n_frame) # remove the overhead
        frame_per_second = n_frame / total_dec_time
        p_fps = frame_per_second
    print (p_frames, p_fps)
    return p_frames, p_fps

if __name__ == "__main__":

    # get passed arguments
    parser = argparse.ArgumentParser(
        description='PyRocDecode Video Decode Arguments')
    parser.add_argument(
        '-i',
        '--input',
        type=str,
        help='Input File Path - required',
        required=True)
    parser.add_argument(
        '-d',
        '--device',
        type=int,
        default=0,
        help='GPU device ID - optional, default 0',
        required=False)
    parser.add_argument(
        '-t',
        '--num_threads',
        type=int,
        default=4,
        help='Num of parallel runs - optional, default 4',
        required=False)
    try:
        args = parser.parse_args()
    except BaseException:
        sys.exit()

    # get params
    input_file_path = args.input
    device_id = args.device
    num_threads = args.num_threads
    sd = 0

    # handle params
    if not os.path.exists(input_file_path):  # Input file (must exist)
        print("ERROR: input file doesn't exist.")
        exit()

    print("info: number of parallel runs: ", num_threads)
    thread_pool = dec.threadpool(num_threads)
    print("info: created a thread pool successfully!")

    # HIP Python calls to find number of VCNs per device
    props = hip.hipDeviceProp_t()
    HipCheck(hip.hipGetDeviceProperties(props, device_id))
    gcn_arch_name = props.gcnArchName.decode('UTF-8')
    gcn_arch_name = gcn_arch_name.split(':', 1)[0]
    num_devices = HipCheck(hip.hipGetDeviceCount())
    if (num_devices < 1):
        print("ERROR: no GPUs found")
        sys.exit()
    if (gcn_arch_name == 'gfx90a' and num_devices > 1):
        sd = 1

    v_device_id  = []
    for i in range(num_threads):
        # use correct device
        if (device_id % 2 == 0):
            if (i % 2 == 0):
                v_device_id.append(device_id)
            else:
                v_device_id.append(device_id + sd)
        else:
            if (i % 2 == 0):
                v_device_id.append(device_id - sd)
            else:
                v_device_id.append(device_id)

    total_frames =  0
    total_fps = 0.0
    #p_frames = [0 for i in range(num_threads)]
    #p_fps = [0.0 for i in range(num_threads)]
    result = []

    for i in range(0, num_threads):
        thread_pool.ExecuteJob(DecProc, input_file_path, v_device_id[i])
    
    print("info: done with executing jobs...waiting to join")
    thread_pool.JoinThreads()

    for thread_id in thread_pool.result.keys():
        result.append(thread_pool.GetThreadResult(thread_id))
    print (result)

    print("info: done with join threads")
    for i in range(0, num_threads):
        total_frames += result[i][0]
        total_fps += result[i][1]

    print("info: Total frame decoded: " + str(total_frames))
    print("info: avg frame per second: " + str(round(total_fps, 2)))