<template>
  <div class="exp-container">
    <div :id="props.canvasContainerId"></div>
  </div>
</template>
<script setup lang="ts">
import { ref, watch } from "vue";
import P5 from "p5";
import type { BBox } from "@/util/types";

const props = defineProps<{
  bgUrl: string;
  bboxes: BBox[];
  scale: number;
  canvasContainerId: string;
}>();
const emit = defineEmits([
  "update:bboxes",
  "update:scale",
  "mouseReleased",
  "mousePressed",
]);
const bboxes = ref<BBox[]>([]);
const scale = ref<number>(1);
const maxCanavasWidth = 800;
const maxCanavasHeight = 500;

watch(bboxes.value, (curVal) => {
  // console.log("bboxes change");

  // Update drawn bboxes to parent component
  emit("update:bboxes", curVal);
});

watch(scale, (curVal) => {
  // Update image scale to parent component
  emit("update:scale", curVal);
});

const sketch = (p5: P5) => {
  let bg: P5.Image | null; // background image
  let startX: number, startY: number; // mouse starting position
  // mouseX and mouseY are the current mouse position (only updated on mouse drag or move (while mouse in canvas boundary))
  let canvasWidth: number, canvasHeight: number, mouseX: number, mouseY: number;
  let originalImgWidth: number, originalImgHeight: number;
  let drawing = false;
  let curBox: BBox | null = null;

  /**
   * General Image Loader
   * @param imageUrl
   */
  const loadImage = (imageUrl: string): Promise<P5.Image> => {
    return new Promise((resolve) => {
      p5.loadImage(imageUrl, (img) => {
        // console.log(`loadImage called, ${img.width} x ${img.height}`);
        resolve(img);
      });
    });
  };

  /**
   * Load Background Image and initialize all related parameters
   * no args is needed, background image url is always props.bgUrl
   */
  const loadBG = async (): Promise<P5.Image> => {
    bg = await loadImage(props.bgUrl); // load image
    // console.log(bg);
    drawing = false;
    curBox = [];
    mouseX = 0;
    mouseY = 0;
    originalImgWidth = bg.width;
    originalImgHeight = bg.height;
    scale.value =
      originalImgWidth > originalImgHeight
        ? maxCanavasWidth / originalImgWidth
        : maxCanavasHeight / originalImgHeight;
    const canvasContainer = document.getElementById(props.canvasContainerId);
    // empty previous canvas
    // while (canvasContainer?.firstChild) {
    //   canvasContainer.removeChild(canvasContainer.lastChild!);
    // }
    canvasWidth = scale.value * originalImgWidth;
    canvasHeight = scale.value * originalImgHeight;
    p5.createCanvas(canvasWidth, canvasHeight).parent(canvasContainer!); // it must be there
    // console.log(`loadImage called, ${originalImgWidth} x ${originalImgHeight}`);
    return bg;
  };

  const mouseInBound = (): boolean =>
    p5.mouseX > 0 &&
    p5.mouseY > 0 &&
    p5.mouseX < canvasWidth &&
    p5.mouseY < canvasHeight;

  watch(
    () => props.bgUrl,
    async (newBgUrl) => {
      // console.log(`load new background image: ${newBgUrl}`);
      bboxes.value.length = 0; // clear all bounding boxes
      curBox = null;
      await loadBG();
    }
  );

  p5.mouseDragged = () => {
    if (drawing && p5.mouseX < canvasWidth && p5.mouseY < canvasHeight) {
      // mouse dragged within canvas
      mouseX = p5.mouseX;
      mouseY = p5.mouseY;
      curBox = [
        Math.min(startX, p5.mouseX),
        Math.min(startY, p5.mouseY),
        Math.abs(p5.mouseX - startX),
        Math.abs(p5.mouseY - startY),
      ];
    }
  };

  p5.mouseMoved = function () {
    // update
    if (mouseInBound()) {
      mouseX = p5.mouseX;
      mouseY = p5.mouseY;
    }
  };

  p5.mousePressed = () => {
    if (mouseInBound()) {
      // mouse pressed within canvas
      drawing = true;
      startX = p5.mouseX;
      startY = p5.mouseY;
      emit("mousePressed");
    }
  };

  p5.mouseReleased = () => {
    if (drawing) {
      // mouse released within canvas
      if (mouseInBound()) {
        // console.log('mouseReleased');

        // draw rectangle
        curBox = null;
        const deltaX = Math.abs(p5.mouseX - startX);
        const deltaY = Math.abs(p5.mouseY - startY);
        if (deltaX * deltaY < 10) {
          // !this is likely a click or a line on the image, will be ignored
          return;
        }
        bboxes.value.push([
          Math.min(startX, p5.mouseX),
          Math.min(startY, p5.mouseY),
          deltaX,
          deltaY,
        ]);
        // console.log(bboxes.value);

        emit("mouseReleased");
      }
      drawing = false;
    }
  };

  p5.setup = () => {
    loadBG();
  };

  p5.draw = () => {
    // draw background image
    if (bg) p5.background(bg);
    // change stroke color to green, weight to 2
    p5.stroke(0, 255, 0);
    p5.strokeWeight(2);
    // change color to white with transparency
    p5.fill(255, 255, 255, 40);
    // draw vertical line
    p5.line(mouseX, 0, mouseX, canvasHeight);
    // draw horizontal line
    p5.line(0, mouseY, canvasWidth, mouseY);
    // change stroke color to red
    p5.stroke(255, 0, 0);
    if (curBox && curBox.length === 4)
      p5.rect(curBox[0], curBox[1], curBox[2], curBox[3]);
    bboxes.value.forEach((bbox) => {
      p5.rect(bbox[0], bbox[1], bbox[2], bbox[3]);
    });
  };
};
new P5(sketch); // invoke p5
</script>
