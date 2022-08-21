<template>
  <div class="exp-container">
    <div :id="props.canvasContainerId"></div>
  </div>
</template>
<script setup lang="ts">
import { ref, watch, onMounted } from "vue";
import P5 from "p5";
import { noiseUrl, noiseTime, bboxTime, imageTime } from "@/util/constant";
import type { BBox } from "@/util/types";

const props = defineProps<{
  bboxes: BBox[];
  bgUrl: string;
  canvasContainerId: string;
}>();
let noiseImg: P5.Image | null = null;
const sketch = (p5: P5) => {
  let bg: P5.Image | null; // background image
  let originalImgWidth: number; // original image's width, used to calculate scale
  let originalImgHeight: number; // original image's height, used to calculate scale
  const maxCanavasWidth = 800; // maximum canvas width
  const maxCanavasHeight = 800; // maximum canvas height
  let scale = ref<number>(1); // image scale after scaling to fit in canvas
  let canvasWidth: number; // actual canvas width to use based on aspect ratio of image
  let canvasHeight: number; // actual canvas height to use based on aspect ratio of image
  const scaledBBoxes = ref<BBox[]>([]);
  const scaledBBoxes2Draw = ref<BBox[]>([]); // actual bounding boxes to display, this is empty when bboxes should be hidden

  const noiseTimer = ref<number | null>(); // timer for displaying the bbox
  const boxTimer = ref<number | null>(); // timer for displaying the image
  const imageTimer = ref<number | null>(); // hide image after timeup

  const clean = () => {
    noiseTimer.value && clearTimeout(noiseTimer.value);
    boxTimer.value && clearTimeout(boxTimer.value);
    imageTimer.value && clearTimeout(imageTimer.value);
  };

  const run = async () => {
    const img = await loadImage(props.bgUrl); // load image
    updateBGProperties(img); // calculate canvas size, create canvas, update all variables and set background to `img`
    loadNoise(); // set background back to noise, TODO: uncomment this
    /**
     * Display a bounding box after the noise
     */
    noiseTimer.value = setTimeout(() => {
      // console.log("load bounding box");
      // display bounding boxes after noise is displayed
      scaledBBoxes2Draw.value = scaledBBoxes.value;
      // console.log(scaledBBoxes2Draw.value);
      /**
       * Load Background Image After the Bbox
       */
      boxTimer.value = setTimeout(() => {
        // console.log("load background image");
        // load background image after noise and bounding boxes are displayed
        // note that this function will resize canvas and recalculate scale
        bg = img;
        /**
         * Display Noise Again
         */
        imageTimer.value = setTimeout(() => {
          // console.log("clear bounding box and display noise again");
          // clear all bounding boxes
          scaledBBoxes.value = [];
          scaledBBoxes2Draw.value = [];
          // display noise again
          loadNoise(); // TODO: uncomment this
        }, imageTime);
      }, bboxTime);
    }, noiseTime);
  };
  watch(
    () => props.bgUrl,
    () => {
      run();
      clean();
      // }
    }
  );
  watch(
    () => props.bboxes,
    (newBBoxes) => {
      // console.log(newBBoxes);
    }
  );

  const loadImage = (imageUrl: string): Promise<P5.Image> => {
    return new Promise((resolve) => {
      p5.loadImage(imageUrl, (img) => {
        // console.log(`loadImage called, ${img.width} x ${img.height}`);
        resolve(img);
      });
    });
  };

  /**
   * Given an image, calculate the canvas size, update the scale and canvas size
   * @param {p5 Image} img
   */
  const updateBGProperties = (img: P5.Image) => {
    // console.log(`updateBG called`);
    // console.log(img);
    originalImgWidth = img.width;
    originalImgHeight = img.height;
    // console.log(
    //   originalImgWidth,
    //   originalImgHeight,
    //   maxCanavasWidth,
    //   maxCanavasHeight
    // );
    scale.value =
      originalImgWidth > originalImgHeight
        ? maxCanavasWidth / originalImgWidth
        : maxCanavasHeight / originalImgHeight;
    // console.log(`scale updated to ${scale.value}`);
    scaledBBoxes.value = props.bboxes.map((bbox) =>
      bbox.map((val: number) => Math.floor(val * scale.value))
    );
    // scaledBBoxes.value = props.bboxes
    // console.warn(`original bboxes: ${props.bboxes}`);
    // console.warn(`scaled bboxes: ${scaledBBoxes.value}`);
    canvasWidth = scale.value * originalImgWidth;
    canvasHeight = scale.value * originalImgHeight;
    // console.warn(`Canvas Size: ${canvasWidth}x${canvasHeight}`);
    const canvasContainer = document.getElementById(props.canvasContainerId);
    while (canvasContainer?.firstChild) {
      canvasContainer.removeChild(canvasContainer.lastChild!);
    }
    p5.createCanvas(canvasWidth, canvasHeight).parent(
      document.getElementById(props.canvasContainerId)!
    ); // it must be there
  };

  /**
   * load noise image and set background to noise
   */
  const loadNoise = () => {
    if (noiseImg !== null) {
      bg = noiseImg;
    } else {
      p5.loadImage(noiseUrl, (img) => {
        noiseImg = img;
        bg = noiseImg;
      });
    }
  };

  p5.setup = async function () {
    await run();
    // console.warn("p5 setup");
    // console.log(bg);
  };

  p5.draw = function () {
    // p5 draw function is only run after setup is finished and bg image loaded
    if (bg) {
      p5.background(bg);
      p5.stroke(255, 255, 0);
      p5.strokeWeight(5);
      p5.fill(255, 255, 255, 0.6);
      scaledBBoxes2Draw.value.forEach((bbox) => {
        p5.rect(bbox[0], bbox[1], bbox[2], bbox[3]);
      });
    }
  };
};

onMounted(() => {
  console.log(props.canvasContainerId);
  new P5(sketch); // invoke p5
});
</script>
