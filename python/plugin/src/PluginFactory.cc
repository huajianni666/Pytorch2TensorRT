#include "NvInferPlugin.h"
#include "NvInfer.h"
#include "NvCaffeParser.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FinetuneLoc.hpp"
#include "FilterBgConf.hpp"
#include "UpSample.hpp"


PYBIND11_MODULE(plugin, m)
{
    namespace py = pybind11;
    // This allows us to use the bindings exposed by the tensorrt module.
    py::module::import("tensorrt");

    m.def("DetectionOutputPlugin", [](bool shareLocation, bool varianceEncodedInTarget,
                                      int backgroundLabelId, int numClasses,
                                      int topK, int keepTopK,
                                      float confidenceThreshold, float nmsThreshold){
        return createNMSPlugin({shareLocation, varianceEncodedInTarget,
                                backgroundLabelId, numClasses,
                                topK, keepTopK,
                                confidenceThreshold, nmsThreshold,
                                nvinfer1::plugin::CodeTypeSSD::CENTER_SIZE, {0 ,1 , 2},
                                false, true});
    }, "detection output");

    py::class_<UpSamplePlugin, nvinfer1::IPluginV2, std::unique_ptr<UpSamplePlugin, py::nodelete>>(m, "UpSamplePlugin")
        // Bind the normal constructor as well as the one which deserializes the plugin
        .def(py::init<std::string, float, int>())
        .def(py::init<std::string, const void*, size_t>())
    ;

    py::class_<FinetuneLocPlugin, nvinfer1::IPluginV2, std::unique_ptr<FinetuneLocPlugin, py::nodelete>>(m, "FinetuneLocPlugin")
        // Bind the normal constructor as well as the one which deserializes the plugin
        .def(py::init<std::string>())
        .def(py::init<std::string, const void*, size_t>())
    ;

    py::class_<FilterBgConfPlugin, nvinfer1::IPluginV2, std::unique_ptr<FilterBgConfPlugin, py::nodelete>>(m, "FilterBgConfPlugin")
        // Bind the normal constructor as well as the one which deserializes the plugin
        .def(py::init<std::string, float>())
        .def(py::init<std::string, const void*, size_t>())
    ;
}
