

with open("embeddings_vgg16.tsv", "r") as input_file:
    with open("embeddings.tsv", "a") as output_file:
        with open("predictions.tsv", "a") as predictions_file:
            i = 0
            while i < 50000:
                output_file.write(input_file.readline())
                predictions_file.write("vgg16\n")
                i += 1


with open("embeddings_resnet.tsv", "r") as input_file:
    with open("embeddings.tsv", "a") as output_file:
        with open("predictions.tsv", "a") as predictions_file:
            i = 0
            while i < 50000:
                output_file.write(input_file.readline())
                predictions_file.write("resnet\n")
                i += 1
