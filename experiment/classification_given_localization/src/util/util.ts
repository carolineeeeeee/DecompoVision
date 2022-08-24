// import {imageUrls, bboxes} from "./constant"
import type { TaskImageInfo, ImageInfo } from "./types";

/**
 * This function is responsible for generating the report for mturk
 * Input elements are created based on the number of imageUrls
 * Each input element will be returned to mturk and reported back to the requester
 * @param {string[]} imageUrls array of image urls
 * @returns
 */
export const initInputElements = (imageUrls: string[]) => {
  // console.log("initInputElements");
  const crowdForm = document.getElementById("custom-crowd-form");
  // console.log(crowdForm);
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
  imageUrls.forEach((url, i) => {
    const classInputEle = document.createElement("input");
    classInputEle.type = "text";
    classInputEle.name = `class-${i}`;
    classInputEle.id = `class-${i}`;
    classInputEle.classList.add("hidden");
    // bboxesInputEle.hidden = true
    form.insertBefore(classInputEle, crowdBtn);
  });
};

// export const initCSVInputElements = () => {

// }

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
