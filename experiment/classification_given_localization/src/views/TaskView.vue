<template>
  <div class="mt-3 flex xl:flex-row lg:flex-col justify-center">
    <div class="flex justify-center">
      <Cavnas
        :bboxes="curBboxes"
        :bgUrl="curBgUrl"
        :canvasContainerId="props.canvasContainerId"
      />
    </div>
    <div class="pl-4">
      <!-- <count-down :exp-time="expTimePerImg" @timeup="imgTimeup" />
      <count-down :exp-time="expTimePerImg" is-progress-bar /> -->
      <div class="grid xl:w-80 lg:w-full lg:grid-cols-10 xl:grid-cols-3 gap-4">
        <el-button
          class="ml-0"
          round
          @click="choose(class_)"
          v-for="(class_, i) in classes"
          :key="i"
          v-show="class_ !== 'other'"
        >
          {{ class_ }}
        </el-button>
      </div>
      <el-button round class="mt-4" type="primary" @click="choose('other')"
        >Other</el-button
      >
      <br />
      <div class="flex justify-center">
        <el-progress
          class="mt-4"
          type="dashboard"
          :percentage="progress * 100"
          :color="colors"
        >
          <template #default="{ percentage }">
            <span class="block text-2xl"
              >{{ (progress * 100).toFixed(2) }}%</span
            >
            <span class="block"
              >{{ Math.min(props.imageData.length, bgImgIdx + 1) }} /
              {{ props.imageData.length }}</span
            >
          </template>
        </el-progress>
      </div>
    </div>
  </div>
</template>
<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import {
  totalWaitingTimePerImage,
  countdownTime,
  classes,
  sanityCheckImages,
} from "@/util/constant";
import CountDown from "@/components/CountDown.vue";
import Cavnas from "@/components/CanvasComponent.vue";
import useResultStore from "@/stores/result";
import { useRouter } from "vue-router";
import type { BBox, ImageInfo, TaskImageInfo } from "../util/types";

const props = defineProps<{
  imageData: ImageInfo[] | TaskImageInfo[];
  mode: "demo" | "task" | "test";
  modelValue?: string[];
  canvasContainerId: string;
}>();

const emit = defineEmits(["update:modelValue"]);

const store = useResultStore();
const router = useRouter();

const expTimePerImg = ref(Date.now() + totalWaitingTimePerImage); // expire time for an image
const expTimePerBox = ref(Date.now() + countdownTime); // expire time for each bounding box

const bgImgIdx = ref(0);
const curBboxes = ref<BBox[]>([]);
const curBgUrl = ref(props.imageData[bgImgIdx.value].url);

const progress = computed(() => (bgImgIdx.value + 1) / props.imageData.length);

const colors = [
  { color: "#f56c6c", percentage: 20 },
  { color: "#e6a23c", percentage: 40 },
  { color: "#5cb87a", percentage: 60 },
  { color: "#1989fa", percentage: 80 },
  { color: "#6f7ad3", percentage: 100 },
];

/**
 * Update variables such as current bounding box url, current bounding boxes and expire time
 * This function is called in nextImage after `bgImgIdx` is incremented
 */
const update = () => {
  // console.log(bgImgIdx.value);
  curBgUrl.value = props.imageData[bgImgIdx.value].url;
  curBboxes.value = props.imageData[bgImgIdx.value].maxAreaBBox!.bbox;
  // console.log(curBboxes);
  expTimePerImg.value = Date.now() + totalWaitingTimePerImage;
  expTimePerBox.value = Date.now() + countdownTime;
};

/**
 * Change to next image and record current image's selected class
 * @param {string} class_
 */
const nextImage = (class_: string) => {
  // if (bgImgIdx.value < props.imageData.length) {
  if (props.modelValue !== undefined)
    emit("update:modelValue", [...props.modelValue, class_]);
  if (props.mode === "task") {
    store.appendClass(class_); // record current image's chosen class
    store.appendImageUrl(curBgUrl.value); // record current image's url
    report(); // this will update the input elements prepared for mturk, only report when in task mode
  }

  const findSanityImg = sanityCheckImages.find(
    (img) => img.url === curBgUrl.value
  );
  if (findSanityImg !== undefined) {
    // current image is a sanity check image
    if (findSanityImg.maxAreaBBox?.name !== class_) {
      // sanity check faild
      // alert(
      //   `sanity check failed, you chose ${class_}, but the correct class is ${findSanityImg.maxAreaBBox?.name}`
      // );
      router.push("/sanity-check-fail");
    }
  } else {
    // not a santity check image, continue
  }
  console.log("debug");
  console.log(findSanityImg);

  bgImgIdx.value += 1; // switch to next image
  if (props.imageData.length <= bgImgIdx.value) {
    // reached end of image list, go to next stage
    console.log("reached end of image list, go to next stage");
    if (props.mode === "task") router.push("/finish");
  } else {
    update();
  }
  // } else {
  // if (props.mode === "task") router.push("/finish");
  // console.warn("No More Images");
  // }
};

/**
 * Image countdown timeup
 */
const imgTimeup = () => {
  console.log("Image Timeup");
};

/**
 * Save user output to input elements so that mturk will pick up these results
 */
const report = () => {
  console.log("report called");
  console.log(store.classes);
  if (props.mode === "task") {
    store.classes.forEach((class_: string, i: number) => {
      const ele: HTMLElement | null = document.getElementById(`class-${i}`);
      if (!ele) return;
      const classInputEle: HTMLInputElement = ele as HTMLInputElement;
      classInputEle.value = class_;
    });
  }
};

/**
 * choose a class for the current image (bounding box)
 * @param {string} class_
 */
const choose = (class_: string) => {
  console.log(`choose ${class_}`);
  if (bgImgIdx.value >= props.imageData.length) {
    return;
  }
  nextImage(class_);
};

onMounted(() => {
  store.clear();
});

// Initialize
update();
</script>
