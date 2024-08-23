/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

#include "roc_video_dec.h"
#include "roc_pydecode.h"
#include "video_post_process.h"

typedef enum ReconfigFlushMode_enum {
    RECONFIG_FLUSH_MODE_NONE = 0,               /**<  Just flush to get the frame count */
    RECONFIG_FLUSH_MODE_DUMP_TO_FILE = 1,       /**<  The remaining frames will be dumped to file in this mode */
    RECONFIG_FLUSH_MODE_CALCULATE_MD5 = 2,      /**<  Calculate the MD5 of the flushed frames */
} ReconfigFlushMode;

// this struct is used by videodecode and videodecodeMultiFiles to dump last frames to file
typedef struct ReconfigDumpFileStruct_t {
    bool b_dump_frames_to_file;
    std::string output_file_name;
} ReconfigDumpFileStruct;

//
// AMD Video Decoder Python Interface class
//
class PyRocVideoDecoder : public RocVideoDecoder {

    public:
        PyRocVideoDecoder(int device_id, int mem_type, rocDecVideoCodec codec, bool force_zero_latency = false,
                          const Rect *p_crop_rect = nullptr, int max_width = 0, int max_height = 0,
                          uint32_t clk_rate = 0) : RocVideoDecoder(device_id, static_cast<OutputSurfaceMemoryType>(mem_type), codec, force_zero_latency,
                          p_crop_rect, false, max_width, max_height, clk_rate) { InitConfigStructure(); }
        ~PyRocVideoDecoder();                        
         
        // for python binding
        int PyDecodeFrame(PyPacketData& packet);
    
        // for python binding
        py::object PyGetFrame(PyPacketData& packet);

        // for python binding
        py::object PyGetFrameRgb(PyPacketData& packet, int rgb_format);

        // for python binding
        uintptr_t PyResizeFrame(PyPacketData& packet, Dim *resized_dim, uintptr_t& in_surf_info); // ret new surface ptr or nullptr

        // for python binding
        uintptr_t PyGetResizedOutputSurfaceInfo();

        // for python binding
        py::object PyReleaseFrame(PyPacketData& packet);
      
        // for python binding
        std::shared_ptr<ConfigInfo> PyGetDeviceinfo();
        
        // for python binding
        py::object PySaveFrameToFile(std::string& output_file_name_in, uintptr_t& surf_mem, uintptr_t& surface_info);

        // for python binding
        py::object PySaveTensorToFile(std::string& output_file_name_in, uintptr_t& surf_mem, int width, int height, int rgb_format, uintptr_t& surf_info);

        // for python binding
        uintptr_t PyGetOutputSurfaceInfo();
 
        // for python binding
        py::object PyGetNumOfFlushedFrames();    

        // for python binding
        py::object PyInitMd5();

        // for python binding
        py::object PyUpdateMd5ForFrame(uintptr_t& surf_mem, uintptr_t& surface_info);

        // for python binding
        py::object PyFinalizeMd5(uintptr_t& digest_back);

        // for python binding
        py::int_ PyGetWidth();

        // for python binding
        py::int_ PyGetHeight();

        // for python binding
        py::int_ PyGetStride();

        // for python binding
        py::int_ PyGetFrameSize();

        // for python binding
        py::object PySetReconfigParams(int flush_mode, std::string& output_file_name_in);
#if OVERHEAD_SUPPORT
        // Session overhead refers to decoder initialization and deinitialization time
        py::object PyAddDecoderSessionOverHead(int session_id, double duration);
        py::object PyGetDecoderSessionOverHead(int session_id);
#endif
    private:
        size_t CalculateRgbImageSize(OutputFormatEnum& e_output_format, OutputSurfaceInfo* p_surf_info);
        std::shared_ptr <ConfigInfo> configInfo;
        void InitConfigStructure();

        // for flush back to support multi-resolution video streams
        ReconfigParams PyReconfigParams;
        ReconfigDumpFileStruct PyReconfigDumpFileStruct;

    protected:
        // used in frame allocation
        uint8_t *frame_ptr_rgb = nullptr;
        VideoPostProcess * post_process_class = nullptr;
        // used in frame resizing
        uint8_t *frame_ptr_resized = nullptr;
        size_t resized_image_size_in_bytes = 0;
        OutputSurfaceInfo *resized_surf_info = nullptr;
};

class PyRocDecThreadPool : public RocDecThreadPool {
    public:
        PyRocDecThreadPool (int nthreads) : RocDecThreadPool(nthreads) {}
        ~PyRocDecThreadPool();
        void PyJoinThreads();
        void PyExecuteJob(std::function<void()> &func);
};