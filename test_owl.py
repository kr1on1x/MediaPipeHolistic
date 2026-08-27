from transformers import Owlv2Processor, Owlv2ForObjectDetection

print("Loading OWLv2...")

processor = Owlv2Processor.from_pretrained(
    "google/owlv2-base-patch16-ensemble"
)

model = Owlv2ForObjectDetection.from_pretrained(
    "google/owlv2-base-patch16-ensemble"
)

print("OWLv2 ONLINE")
