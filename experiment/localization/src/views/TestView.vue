<template>
  <div class="pt-5">
    <p>
      Complete this qualification test for only once. You will receive a skip
      code so you won't need to do this again for another HIT.
    </p>
    <el-input
      v-model="codeInput"
      placeholder="Please Skip Code If You Have One"
    />
    <el-button @click="check" class="mt-2 float-right">Check</el-button>
    <task-view
      :imageUrls="testImageUrls"
      :countdownTime="countdownTime"
      :bboxTimeouts="testImageUrls.map(() => countdownTime)"
      :bboxImageTimeouts="testImageUrls.map(() => totalWaitingTimePerImage)"
      @done="onDone"
      @next-image="onNextImage"
      mode="test"
      canvas-container-id="test-canvas-container"
    />

    <h3 class="text-3xl">Result</h3>
    <div v-if="passed === null">
      <p>Test In Progress</p>
    </div>
    <div v-else>
      <div v-if="passed === false">
        <h3 class="text-2xl">Failed</h3>
        <p>
          You didn't pass the quantitative test, please quit the job and try
          again.
        </p>
      </div>
      <div v-else-if="passed === true">
        <h3 class="text-2xl">Passed</h3>
        <p>Congratulations, you passed the qualification test</p>
        <p>
          Here is the skip code: <strong>{{ qualificationCode }}</strong>
        </p>
        <p>
          <strong>Please Take Note of the code</strong> so that you don't need
          to redo this test next time.
        </p>
        <el-button
          class="mb-5 float-right"
          size="large"
          type="primary"
          @click="proceed"
          >Proceed</el-button
        >
      </div>
      <p>Your Score is {{ (score * 100).toFixed(2) }}%</p>
    </div>
  </div>
</template>
<script setup lang="ts">
import TaskView from "./TaskView.vue";
import { ref, computed, watch, onMounted } from "vue";
import { useRouter } from "vue-router";
import {
  qualificationTestData,
  qualificationCode,
  imageUrls,
  countdownTime,
  bboxTimeouts,
  totalWaitingTimePerImage,
  qualificationTestPassThreshold,
  bboxImageTimeouts,
  goodIouThreshold,
} from "@/util/constant";
import useResultStore from "@/stores/result";
import { iou } from "@/util/util";

const passed = ref<null | boolean>(null);
const score = ref(0); // percentage of passed images
const store = useResultStore();
const gtMaxAreaBboxes = qualificationTestData.map((d) => d.maxAreaBBox?.bbox);

const evaluate = () => {
  let countGoodIou = 0;
  Object.entries(store.bboxes).forEach(([url, payload]) => {
    // find gt
    const foundData = qualificationTestData.find((d) => d.url === url);
    if (foundData === undefined)
      throw new Error("Unexpected Error: URL Not Found");
    const ious = payload.bboxes.map((bbox) => {
      if (foundData.maxAreaBBox == undefined)
        throw new Error(
          `Unexpected Error: Image ${foundData.url} has no maxAreaBBox`
        );
      return iou(
        [
          bbox[0] / payload.scale,
          bbox[1] / payload.scale,
          bbox[2] / payload.scale,
          bbox[3] / payload.scale,
        ],
        foundData.maxAreaBBox.bbox[0]
      );
    });
    const maxIou = Math.max(...ious);
    console.log(`maxIou: ${maxIou}`);
    if (maxIou >= goodIouThreshold) countGoodIou++;
  });
  return countGoodIou;
};

const testImageUrls = qualificationTestData.map((d) => d.url);

const onDone = () => {
  console.log("onDone");
  const numPassed = evaluate();
  console.log(`numPassed: ${numPassed}`);
  score.value = numPassed / qualificationTestData.length;
  passed.value = score.value >= qualificationTestPassThreshold;
};

const onNextImage = () => {
  console.log("onNextImage");
  const numPassed = evaluate();
  Object.entries(store.bboxes).forEach(([url, payload]) => {
    if (payload.bboxes.length > 10) {
      passed.value = false;
    }
  });
  console.log(`numPassed: ${numPassed}`);
};

const router = useRouter();
const codeInput = ref("");
const check = () => {
  if (qualificationCode === codeInput.value) {
    router.push("/task");
  } else {
    alert("Wrong Skip Code");
  }
};

const proceed = () => {
  router.push("/task");
};
</script>
