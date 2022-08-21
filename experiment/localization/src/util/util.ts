import type { BBox, ImageInfo, TaskImageInfo } from "./types";

export const initInputElements = (imageUrls: string[]) => {
  console.log("initInputElements");

  const crowdForm = document.getElementById("custom-crowd-form");
  if (!crowdForm) {
    return setTimeout(() => {
      initInputElements(imageUrls);
    }, 1000);
  }
  const inputEles = crowdForm.querySelectorAll("input");
  inputEles.forEach((ele) => ele.remove());
  const form = Array.from(crowdForm.getElementsByTagName("form"))[0];
  if (!form) {
    return setTimeout(() => {
      initInputElements(imageUrls);
    }, 1000);
  }
  const crowdBtn = form.children[0];
  console.log(crowdBtn);

  imageUrls.forEach((url, i) => {
    const urlInputEle = document.createElement("input");
    urlInputEle.type = "text";
    urlInputEle.name = `url-${i}`;
    urlInputEle.id = `url-${i}`;
    urlInputEle.hidden = true;
    form.insertBefore(urlInputEle, crowdBtn);
    const scaleInputEle = document.createElement("input");
    scaleInputEle.type = "text";
    scaleInputEle.name = `scale-${i}`;
    scaleInputEle.id = `scale-${i}`;
    scaleInputEle.hidden = true;
    form.insertBefore(scaleInputEle, crowdBtn);
    const bboxesInputEle = document.createElement("input");
    bboxesInputEle.type = "text";
    bboxesInputEle.name = `bbox-${i}`;
    bboxesInputEle.id = `bbox-${i}`;
    bboxesInputEle.hidden = true;
    form.insertBefore(bboxesInputEle, crowdBtn);
  });
};

export function randomIntFromInterval(min: number, max: number) {
  return Math.floor(Math.random() * (max - min + 1) + min);
}

export const fillLargestBBox = (imageData: ImageInfo[]) => {
  imageData.forEach((image) => {
    const areas = image.objects.map(
      (object) => object.bbox[2] * object.bbox[3]
    );
    const maxArea = Math.max(...areas);
    const maxAreaIndex = areas.indexOf(maxArea);
    image.maxAreaBBox = {
      bbox: [image.objects[maxAreaIndex].bbox],
      name: image.objects[maxAreaIndex].name,
    };
  });
};

export const fillLargestTaskBBox = (imageData: TaskImageInfo[]) => {
  imageData.forEach((image) => {
    const areas = image.objects.map(
      (object) => object.bbox[2] * object.bbox[3]
    );
    const maxArea = Math.max(...areas);
    const maxAreaIndex = areas.indexOf(maxArea);
    image.maxAreaBBox = {
      bbox: [image.objects[maxAreaIndex].bbox],
    };
  });
};

export const iou = (bbox1: BBox, bbox2: BBox) => {
  const bbox1Transformed = [
    bbox1[0],
    bbox1[1],
    bbox1[0] + bbox1[2],
    bbox1[1] + bbox1[3],
  ];
  const bbox2Transformed = [
    bbox2[0],
    bbox2[1],
    bbox2[0] + bbox2[2],
    bbox2[1] + bbox2[3],
  ];

  const xA = Math.max(bbox1Transformed[0], bbox2Transformed[0]);
  const yA = Math.max(bbox1Transformed[1], bbox2Transformed[1]);
  const xB = Math.min(bbox1Transformed[2], bbox2Transformed[2]);
  const yB = Math.min(bbox1Transformed[3], bbox2Transformed[3]);
  // compute the area of intersection rectangle
  const interArea = Math.max(0, xB - xA + 1) * Math.max(0, yB - yA + 1);
  // compute the area of both the prediction and ground-truth
  // rectangles
  const boxAArea =
    (bbox1Transformed[2] - bbox1Transformed[0] + 1) *
    (bbox1Transformed[3] - bbox1Transformed[1] + 1);
  const boxBArea =
    (bbox2Transformed[2] - bbox2Transformed[0] + 1) *
    (bbox2Transformed[3] - bbox2Transformed[1] + 1);
  // compute the intersection over union by taking the intersection
  // area and dividing it by the sum of prediction + ground-truth
  // areas - the interesection area
  const iou = interArea / (boxAArea + boxBArea - interArea);
  // return the intersection over union value
  return Math.max(iou, 0);
};
