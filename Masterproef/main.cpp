// main.cpp
#include <iostream>
#include <vector>
#include <onnxruntime_cxx_api.h>

int main() {
    try {
        // 1) Env en session opties
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "masterproef");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

        // 2) Model path (vervang door jouw .onnx model)
        const wchar_t* model_path = L"bert-base-cased.onnx"; // voorbeeld
        Ort::Session session(env, model_path, session_options);
        std::cout << "Model geladen.\n";

        // 3) Voorbeeld: input/output naam ophalen
        Ort::AllocatorWithDefaultOptions allocator;
        char* in_name = session.GetInputNameAllocated(0, allocator).get();
        char* out_name = session.GetOutputNameAllocated(0, allocator).get();
        std::cout << "Input name: " << in_name << "\n";
        std::cout << "Output name: " << out_name << "\n";

        // 4) Dummy input (voorbeeld: int64 tensor met shape [1,8])
        std::vector<int64_t> input_shape = { 1, 8 };
        std::vector<int64_t> input_data = { 101, 7592, 2003, 1037, 3076, 102, 0, 0 }; // token ids voorbeeld
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(mem_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());

        const char* input_names[] = { in_name };
        const char* output_names[] = { out_name };

        // 5) Run inference
        auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_names, &input_tensor, 1, output_names, 1);

        // 6) Lees resultaat (voorbeeld: float outputs)
        if (!output_tensors.empty() && output_tensors.front().IsTensor()) {
            float* out_data = output_tensors.front().GetTensorMutableData<float>();
            size_t out_elems = 1;
            for (size_t i = 0; i < output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount(); ++i) {
                std::cout << "out[" << i << "] = " << out_data[i] << "\n";
                if (i > 20) { std::cout << "... (truncated)\n"; break; }
            }
        }

        std::cout << "Klaar\n";
    }
    catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
    }
    catch (const std::exception& ex) {
        std::cerr << "Exception: " << ex.what() << std::endl;
    }
    return 0;
}
