<template>
  <h1 class="text-4xl font-bold">Task</h1>
  <div class="flex">
    <div class="canvas-container w-[800px] flex justify-center">
      <Canvas
        :bgUrl="bgUrl"
        v-model:bboxes="bboxes"
        v-model:scale="scale"
        @mouseReleased="mouseReleased"
        @mousePressed="mousePressed"
        :canvas-container-id="props.canvasContainerId"
      />
    </div>
    <div class="count-down-container w-80 ml-4">
      <h2 class="text-xl font-semibold">Image Countdown</h2>
      <count-down :exp-time="expTimePerImg" @timeup="imageTimeup" />
      <count-down :exp-time="expTimePerImg" is-progress-bar />
      <h2 class="text-xl font-semibold">Bounding Box Countdown</h2>
      <count-down :exp-time="expTimePerBox" @timeup="bboxTimeup" />
      <count-down :exp-time="expTimePerBox" is-progress-bar />
      <h2 class="text-xl font-semibold">
        Progress:
        <span class="font-normal"
          >Image {{ `${bgImgIdx + 1} / ${imageUrls.length}` }}</span
        >
      </h2>
      <el-progress
        :percentage="Math.round((bgImgIdx / imageUrls.length) * 100)"
      />
      <el-button v-if="isDev" @click="nextImage">Next Image</el-button>
    </div>
  </div>
</template>
<script setup lang="ts">
import { ref, onMounted } from "vue";
import useResultStore from "@/stores/result";
import { useRouter } from "vue-router";
import Canvas from "@/components/CanvasComponent.vue";
import type { BBox } from "@/util/types";
import {
  sanityCheckData,
  qualificationTestPassThreshold,
} from "@/util/constant";
import { iou } from "@/util/util";

const isDev = import.meta.env.DEV;

const props = defineProps<{
  imageUrls: string[];
  countdownTime: number;
  bboxTimeouts: number[];
  bboxImageTimeouts: number[];
  mode: "demo" | "task" | "test";
  canvasContainerId: string;
}>();

const emit = defineEmits(["update:modelValue", "done", "nextImage"]);

const store = useResultStore();
const router = useRouter();

const bgImgIdx = ref(0);

const bboxes = ref([]);
const scale = ref(1);

let bgUrl = ref(props.imageUrls[bgImgIdx.value]);
let expTimePerImg = ref(Date.now() + props.bboxImageTimeouts[bgImgIdx.value]); // expire time for an image
let expTimePerBox = ref(Date.now() + props.bboxTimeouts[bgImgIdx.value]); // expire time for each bounding box
const initVars = () => {
  bboxes.value = [];
  // scale.value = 2;
  if (bgImgIdx.value === props.imageUrls.length) {
    console.warn("initVars, index out of bound (handled)");
    return;
  }
  bgUrl.value = props.imageUrls[bgImgIdx.value];
  expTimePerImg.value = Date.now() + props.bboxImageTimeouts[bgImgIdx.value];
  expTimePerBox.value = Date.now() + props.bboxTimeouts[bgImgIdx.value];
};

/**
 * Refresh Bounding Box Countdown when mouse is released
 */
const mouseReleased = () => {
  expTimePerBox.value = Date.now() + props.bboxTimeouts[bgImgIdx.value];
};

const mousePressed = () => {
  // console.log("mousePressed");
};

const report = () => {
  const keys = Object.keys(store.bboxes); // image urls as key
  for (let i = 0; i < keys.length; i++) {
    const url = keys[i];
    const value = store.bboxes[url];
    const bboxesStr = JSON.stringify(value.bboxes);
    const urlInputEle = document.getElementById(`url-${i}`);
    if (urlInputEle) (urlInputEle as HTMLInputElement).value = url;
    const scaleInputEle = document.getElementById(`scale-${i}`);
    if (scaleInputEle)
      (scaleInputEle as HTMLInputElement).value = value.scale.toString();
    const bboxesInputEle = document.getElementById(`bbox-${i}`);
    if (bboxesInputEle)
      (bboxesInputEle as HTMLInputElement).value = JSON.stringify(bboxesStr);
  }
};

const nextImage = () => {
  emit("nextImage");
  console.log("next image");

  // check if current image is a sanity check image
  const findSanityImage = sanityCheckData.find((d) => d.url === bgUrl.value);
  if (findSanityImage === undefined) {
    // not sanity check image, continue
  } else {
    // is sanity check image, check if passed
    if (findSanityImage.maxAreaBBox == undefined) {
      throw new Error(
        `Unexpected Error: ${findSanityImage.url} has no maxAreaBBox`
      );
    } else {
      const ious = Array.from(bboxes.value).map((bbox) =>
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        iou(findSanityImage.maxAreaBBox!.bbox[0], [
          bbox[0] / scale.value,
          bbox[1] / scale.value,
          bbox[2] / scale.value,
          bbox[3] / scale.value,
        ])
      );
      console.log(ious);
      const maxIou = Math.max(...ious);
      console.log(maxIou);
      console.log(maxIou < qualificationTestPassThreshold);
      if (maxIou < qualificationTestPassThreshold) {
        // alert("Sanity Check Failed");
        router.push("/sanity-check-fail");
      }
    }
    // const maxAreaBbox = findSanityImage.maxAreaBBox.bbox
  }
  if (bgImgIdx.value < props.imageUrls.length) {
    store.addImageBBox({
      scale: scale.value,
      bboxes: Array.from(bboxes.value),
      url: props.imageUrls[bgImgIdx.value],
    });
    bgImgIdx.value += 1;
    if (props.mode === "task") report();
  }

  if (props.imageUrls.length === bgImgIdx.value) {
    emit("done");
    if (props.mode === "task") {
      router.push("/finish");
    } else if (props.mode === "test") {
      // router.push("/test");
    }
  } else {
    initVars();
  }
};

// Handle Countdown Time Is Up
const imageTimeup = () => nextImage();
const bboxTimeup = () => nextImage();

// handle undo
const keyDownMap = new Map();
document.addEventListener("keydown", (e) => {
  keyDownMap.set(e.key, true);
  if (keyDownMap.get("Control") && keyDownMap.get("z")) {
    bboxes.value.pop();
    expTimePerBox.value = Date.now() + props.countdownTime;
  }
});
document.addEventListener("keyup", (e) => {
  keyDownMap.delete(e.key);
});

onMounted(() => {
  store.clear();
  // console.log(`TaskView OnMounted`);
  // console.log(props.imageUrls);
  // console.log(props.bboxImageTimeouts);
  // console.log(props.bboxTimeouts);

  const customCrowdForm = document.getElementById("custom-crowd-form");
  console.log(customCrowdForm);

  customCrowdForm?.classList.add("hidden");
});
</script>
