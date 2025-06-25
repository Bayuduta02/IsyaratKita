from tflite_support import metadata as _metadata
from tflite_support import metadata_schema_py_generated as _metadata_fb
from tflite_support import flatbuffers

# ----- Buat metadata -----
model_meta = _metadata_fb.ModelMetadataT()
model_meta.name = "Ultralytics BISINDO Detector"
model_meta.description = "Model untuk mendeteksi gestur BISINDO a–z"
model_meta.version = "8.3.158"
model_meta.author = "Bayu"
model_meta.license = "AGPL-3.0"

# ----- Input (gambar RGB) -----
input_meta = _metadata_fb.TensorMetadataT()
input_meta.name = "image"
input_meta.description = "Input gambar RGB ukuran 640x640"
input_meta.content = _metadata_fb.ContentT()
input_meta.content.contentPropertiesType = _metadata_fb.ContentProperties.ImageProperties
input_meta.content.contentProperties = _metadata_fb.ImagePropertiesT()
input_meta.content.contentProperties.colorSpace = _metadata_fb.ColorSpaceType.RGB

# ----- Output (deteksi gestur) -----
output_meta = _metadata_fb.TensorMetadataT()
output_meta.name = "gestures"
output_meta.description = "Hasil deteksi gestur BISINDO"
output_meta.content = _metadata_fb.ContentT()
output_meta.content.contentPropertiesType = _metadata_fb.ContentProperties.FeatureProperties
output_meta.content.contentProperties = _metadata_fb.FeaturePropertiesT()

label_file = _metadata_fb.AssociatedFileT()
label_file.name = "labels.txt"
label_file.description = "Label BISINDO a–z"
label_file.type = _metadata_fb.AssociatedFileType.TENSOR_AXIS_LABELS
output_meta.associatedFiles = [label_file]

# ----- Gabungkan menjadi subgraph -----
subgraph = _metadata_fb.SubGraphMetadataT()
subgraph.inputTensorMetadata = [input_meta]
subgraph.outputTensorMetadata = [output_meta]
model_meta.subgraphMetadata = [subgraph]

# ----- Serialize menjadi metadata.bin -----
b = flatbuffers.Builder(0)
b.Finish(model_meta.Pack(b), _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
metadata_buf = b.Output()
with open("metadata.bin", "wb") as f:
    f.write(metadata_buf)

# ----- Sisipkan ke model TFLite -----
model_file = "best_float32.tflite"  # Ganti kalau nama modelnya beda
populator = _metadata.MetadataPopulator.with_model_file(model_file)
populator.load_metadata_buffer(metadata_buf)
populator.load_associated_files(["labels.txt"])
populator.populate()

print("✅ Metadata berhasil ditambahkan ke", model_file)
