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
      :image-data="qualificationTestData"
      mode="test"
      v-model="resultClasses"
      canvasContainerId="test-canvas-container"
    />

    <h3 class="text-2xl">Your Choices</h3>
    <p>{{ resultClasses }}</p>

    <div v-if="resultClasses.length === ans.length">
      <p>Match: {{ compareAns }}</p>
      <h3 class="text-2xl"># of Correct</h3>
      <p>
        {{ numMatch }} /
        {{ compareAns.length }}
      </p>

      <div v-if="numMatch === ans.length">
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
      <div v-else>
        <p>
          You didn't pass the quantitative test, please quit the job and try
          again.
        </p>
      </div>
    </div>
  </div>
</template>
<script setup lang="ts">
import TaskView from "./TaskView.vue";
import { qualificationTestData, qualificationCode } from "../util/constant";
import { ref, computed, watch, onMounted } from "vue";
import { useRouter } from "vue-router";

const ans = qualificationTestData.map((img) => img.maxAreaBBox?.name);
const resultClasses = ref([] as string[]);
const compareAns = computed(() =>
  ans.map((a, i) => a === resultClasses.value[i])
);

const numMatch = computed(() =>
  (compareAns.value.map((x) => (x ? 1 : 0)) as number[]).reduce(
    (a, b) => a + b,
    0
  )
);

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

const sumArr = (arr: number[]) =>
  arr.reduce((partialSum, a) => partialSum + a, 0);

onMounted(() => {
  console.log(qualificationTestData);
});
</script>
