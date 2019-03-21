#include "FullyConnected.h"
#include "UpSample.h"
#include "DetectionOutput.h"
#include "Normalize.h"
#include "PriorBox.h"
#include "FilterBgConf.h"
#include "FinetuneLoc.h"
#include "NvInferPlugin.h"
#include "NvInfer.h"
#include "NvCaffeParser.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

PYBIND11_MODULE(plugin, m)
{
    namespace py = pybind11;

    // This allows us to use the bindings exposed by the tensorrt module.
    py::module::import("tensorrt");

    // Note that we only need to bind the constructors manually. Since all other methods override IPlugin functionality, they will be automatically available in the python bindings.
    // The `std::unique_ptr<FCPlugin, py::nodelete>` specifies that Python is not responsible for destroying the object. This is required because the destructor is private.
    py::class_<FCPlugin, nvinfer1::IPluginExt, std::unique_ptr<FCPlugin, py::nodelete>>(m, "FCPlugin")
        // Bind the normal constructor as well as the one which deserializes the plugin
	.def(py::init<const nvinfer1::Weights*, int>())
	.def(py::init<const nvinfer1::Weights*, const nvinfer1::Weights*>())
        .def(py::init<const void*, size_t>())
    ;

    // Since the createPlugin function overrides IPluginFactory functionality, we do not need to explicitly bind it here.
    // We specify py::multiple_inheritance because we have not explicitly specified nvinfer1::IPluginFactory as a base class.
    py::class_<FCPluginFactory, nvcaffeparser1::IPluginFactoryExt>(m, "FCPluginFactory", py::multiple_inheritance())
        // Bind the default constructor.
        .def(py::init<>())
        // The destroy_plugin function does not override the base class, so we must bind it explicitly.
        .def("destroy_plugin", &FCPluginFactory::destroyPlugin)
    ;
    
    m.def("PriorBoxPlugin", &PriorBoxPlugin, "create priorbox");
 
    m.def("DetectionOutputPlugin", &DetectionOutputPlugin, "detection output");
    
    m.def("NormalizePlugin", &NormalizePlugin, "create normalize");
    
    py::class_<UpSamplePlugin, nvinfer1::IPluginExt, std::unique_ptr<UpSamplePlugin, py::nodelete>>(m, "UpSamplePlugin")
        // Bind the normal constructor as well as the one which deserializes the plugin
        .def(py::init<float, int>())
        .def(py::init<const void*, size_t>())
    ;

    py::class_<UpSampleFactory, nvcaffeparser1::IPluginFactoryExt>(m, "UpSampleFactory", py::multiple_inheritance())
        // Bind the default constructor.
        .def(py::init<>())
        // The destroy_plugin function does not override the base class, so we must bind it explicitly.
        .def("destroy_plugin", &UpSampleFactory::destroyPlugin)
    ;

    py::class_<FinetuneLocPlugin, nvinfer1::IPluginExt, std::unique_ptr<FinetuneLocPlugin, py::nodelete>>(m, "FinetuneLocPlugin")
        // Bind the normal constructor as well as the one which deserializes the plugin
        .def(py::init<>())
        .def(py::init<const void*, size_t>())
    ;

    py::class_<FinetuneLocPluginFactory, nvcaffeparser1::IPluginFactoryExt>(m, "FinetuneLocPluginFactory", py::multiple_inheritance())
        // Bind the default constructor.
        .def(py::init<>())
        // The destroy_plugin function does not override the base class, so we must bind it explicitly.
        .def("destroy_plugin", &FinetuneLocPluginFactory::destroyPlugin)
    ;

    py::class_<FilterBgConfPlugin, nvinfer1::IPluginExt, std::unique_ptr<FilterBgConfPlugin, py::nodelete>>(m, "FilterBgConfPlugin")
        // Bind the normal constructor as well as the one which deserializes the plugin
        .def(py::init<float>())
        .def(py::init<const void*, size_t>())
    ;

    py::class_<FilterBgConfPluginFactory, nvcaffeparser1::IPluginFactoryExt>(m, "FilterBgConfPluginFactory", py::multiple_inheritance())
        // Bind the default constructor.
        .def(py::init<>())
        // The destroy_plugin function does not override the base class, so we must bind it explicitly.
        .def("destroy_plugin", &FilterBgConfPluginFactory::destroyPlugin)
    ;    
}
