"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import os
import cv2

import numpy as np
from PIL import Image

from synthtiger import components, layers, templates


class Multiline(templates.Template):
    def __init__(self, config=None):
        if config is None:
            config = {}

        self.count = config.get("count", 100)
        self.paragraphs = config.get("paragraphs", 5)
        self.doc_size = config.get("doc_size", (1024, 1024))

        self.corpus = components.BaseCorpus(**config.get("corpus", {}))
        self.font = components.BaseFont(**config.get("font", {}))
        self.color = components.Gray(**config.get("color", {}))
        self.layout = components.FlowLayout(**config.get("layout", {}))
        self.doc_layout = components.FlowLayout(**config.get("doc_layout", {}))

        self.transform = components.Switch(
            components.Selector(
                [
                    components.Perspective(),
                    components.Perspective(),
                    components.Trapezoidate(),
                    components.Trapezoidate(),
                    components.Skew(),
                    components.Skew(),
                    components.Rotate(),
                ]
            ),
            **config.get("transform", {}),
        )

        self.style = components.Switch(
            components.Selector(
                [
                    components.TextBorder(),
                    components.TextShadow(),
                    components.TextExtrusion(),
                ]
            ),
            **config.get("style", {}),
        )

        self.shape = components.Switch(
            components.Selector(
                [components.ElasticDistortion(), components.ElasticDistortion()]
            ),
            **config.get("shape", {}),
        )

    def generate_paragraph(self):
        num_texts = np.random.randint(1, self.count+1)
        texts = [self.corpus.data(self.corpus.sample()) for _ in range(num_texts)]
        font = self.font.sample()
        style = self.style.sample()
        color = self.color.data(self.color.sample())
        transform = self.transform.sample()

        text_layers = [
            layers.TextLayer(text, color=color, **font) for text in texts
        ]

        self.style.apply(text_layers, style)
        self.shape.apply(text_layers)
        self.layout.apply(text_layers)

        #text_group = layers.Group(text_layers)
        text_group = layers.Group(text_layers).merge()
        #self.layout.apply(text_group)

        #self.style.apply(text_group, style)
        #self.layout.apply(text_group)

        self.transform.apply(
            [text_group, *text_layers], transform
        )
        
        for layer in text_layers:
            layer.topleft -= text_group.topleft

        #image = text_group.output()
        #
        #boxes = [
        #    list(map(int, layer.bbox)) for layer in text_layers
        #]

        #quads = [
        #
        #]

        #for layer in text_layers:
        #    quad = layer.quad
        #    cv2.polylines(image, [np.array(quad, np.int32).reshape((-1, 1, 2))], True, (255, 255, 255), 2)    
        #cv2.imwrite("test.jpg", image)
        #for box in boxes:
        #    x, y, w, h = box
        #    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 255), 2)

        #cv2.imwrite("test.jpg", image)

        #label = " ".join(texts)

        #data = {
        #    "image": image,
        #    "texts": texts,
        #    "boxes": boxes,
        #    "label": label,
        #}
        #return data
            
        return text_group, text_layers

    def generate(self):
        bg_doc = layers.RectLayer(self.doc_size, (255, 255, 255, 255))

        paragraph_layers = []
        text_layers = []

        num_paragraph = np.random.randint(1, self.paragraphs+1)
        for _ in range(num_paragraph):
            #paragraph = self.generate_paragraph()
            paragraph, text_layer = self.generate_paragraph()
            paragraph_layers.append(paragraph)
            text_layers.extend(text_layer)

        doc_group = layers.Group(paragraph_layers)
        self.doc_layout.apply(doc_group)

        #image = (doc_group + bg_doc).output()
        #cv2.imwrite("test.png", doc_group.output())

        image = doc_group.output()
        for layer in text_layers:
            quad = layer.quad
            cv2.polylines(image, [np.array(quad, np.int32).reshape((-1, 1, 2))], True, (255, 255, 255), 2)    
        cv2.imwrite("test.jpg", image)
            
        return {
            "image": doc_group.output(),
            "label": "dummy"
        }
    
    def init_save(self, root):
        os.makedirs(root, exist_ok=True)
        gt_path = os.path.join(root, "gt.txt")
        self.gt_file = open(gt_path, "w", encoding="utf-8")

    def save(self, root, data, idx):
        image = data["image"]
        label = data["label"]

        shard = str(idx // 10000)
        image_key = os.path.join("images", shard, f"{idx}.jpg")
        image_path = os.path.join(root, image_key)

        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        image = Image.fromarray(image[..., :3].astype(np.uint8))
        image.save(image_path, quality=95)

        self.gt_file.write(f"{image_key}\t{label}\n")

    def end_save(self, root):
        self.gt_file.close()
