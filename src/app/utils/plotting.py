# app/utils/plotting.py

from typing import List, Optional, Dict
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import random
import plotly.express as px
import plotly.graph_objects as go

from .image_ops import mask_to_polygon
from .results import DetectionResult

def annotate(image: Image.Image, detection_results: List[DetectionResult]) -> np.ndarray:
    image_cv2 = np.array(image) if isinstance(image, Image.Image) else image
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

    for detection in detection_results:
        label = detection.label
        score = detection.score
        box = detection.box
        mask = detection.mask

        color = np.random.randint(0, 256, size=3)

        cv2.rectangle(image_cv2, (box.xmin, box.ymin), (box.xmax, box.ymax), color.tolist(), 2)
        cv2.putText(image_cv2, f"{label}: {score:.2f}", (box.xmin, box.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)

        if mask is not None:
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image_cv2, contours, -1, color.tolist(), 2)

    return cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

def plot_detections(image: Image.Image, detections: List[DetectionResult], save_name: Optional[str] = None) -> None:
    annotated_image = annotate(image, detections)
    # if save_name:
    #     plt.imshow(annotated_image)
    #     plt.axis('off')
    #     plt.savefig(save_name, bbox_inches='tight')
    #     plt.close()

def random_named_css_colors(num_colors: int) -> List[str]:
    named_css_colors = [
        'aliceblue','antiquewhite','aqua','aquamarine','azure','beige','bisque','black','blanchedalmond',
        'blue','blueviolet','brown','burlywood','cadetblue','chartreuse','chocolate','coral','cornflowerblue',
        'cornsilk','crimson','cyan','darkblue','darkcyan','darkgoldenrod','darkgray','darkgreen','darkgrey',
        'darkkhaki','darkmagenta','darkolivegreen','darkorange','darkorchid','darkred','darksalmon','darkseagreen',
        'darkslateblue','darkslategray','darkslategrey','darkturquoise','darkviolet','deeppink','deepskyblue',
        'dimgray','dimgrey','dodgerblue','firebrick','floralwhite','forestgreen','fuchsia','gainsboro','ghostwhite',
        'gold','goldenrod','gray','green','greenyellow','grey','honeydew','hotpink','indianred','indigo','ivory',
        'khaki','lavender','lavenderblush','lawngreen','lemonchiffon','lightblue','lightcoral','lightcyan','lightgoldenrodyellow',
        'lightgray','lightgreen','lightgrey','lightpink','lightsalmon','lightseagreen','lightskyblue','lightslategray',
        'lightslategrey','lightsteelblue','lightyellow','lime','limegreen','linen','magenta','maroon','mediumaquamarine',
        'mediumblue','mediumorchid','mediumpurple','mediumseagreen','mediumslateblue','mediumspringgreen','mediumturquoise',
        'mediumvioletred','midnightblue','mintcream','mistyrose','moccasin','navajowhite','navy','oldlace','olive',
        'olivedrab','orange','orangered','orchid','palegoldenrod','palegreen','paleturquoise','palevioletred','papayawhip',
        'peachpuff','peru','pink','plum','powderblue','purple','rebeccapurple','red','rosybrown','royalblue','saddlebrown',
        'salmon','sandybrown','seagreen','seashell','sienna','silver','skyblue','slateblue','slategray','slategrey',
        'snow','springgreen','steelblue','tan','teal','thistle','tomato','turquoise','violet','wheat','white',
        'whitesmoke','yellow','yellowgreen'
    ] # (list panjang warna)
    return random.sample(named_css_colors, min(num_colors, len(named_css_colors)))

def plot_detections_plotly(image: np.ndarray, detections: List[DetectionResult], class_colors: Optional[Dict[str, str]] = None) -> None:
    if class_colors is None:
        num_detections = len(detections)
        colors = random_named_css_colors(num_detections)
        class_colors = {i: colors[i] for i in range(num_detections)}

    fig = px.imshow(image)

    shapes = []
    for idx, detection in enumerate(detections):
        box = detection.box
        polygon = mask_to_polygon(detection.mask)
        fig.add_trace(go.Scatter(
            x=[p[0] for p in polygon] + [polygon[0][0]],
            y=[p[1] for p in polygon] + [polygon[0][1]],
            mode="lines",
            line=dict(color=class_colors[idx], width=2),
            fill="toself",
            name=f"{detection.label}: {detection.score:.2f}"
        ))

        shape = [dict(
            type="rect",
            xref="x", yref="y",
            x0=box.xmin, y0=box.ymin,
            x1=box.xmax, y1=box.ymax,
            line=dict(color=class_colors[idx])
        )]
        shapes.append(shape)

    fig.update_layout(showlegend=True)
    fig.show()



