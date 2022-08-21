<script setup lang="ts">
import { useRouter } from "vue-router";
import TaskView from "./TaskView.vue";
import { qualificationTestData, qualificationCode } from "@/util/constant";
import { ref } from "vue";

const codeInput = ref();

const router = useRouter();
const proceed = () => {
  router.push("/test");
};

const check = () => {
  if (qualificationCode === codeInput.value) {
    router.push("/task");
  } else {
    alert("Wrong Skip Code");
  }
};
</script>

<template>
  <main>
    <!-- Instruction -->
    <h2 class="text-2xl">Instructions</h2>
    <p>
      In this job, you are asked to select the class of the object in each image
      from the given list of choices.
    </p>
    <p>
      Each task is timed and you can look at the picture for just 200 ms before
      it disappear so stay focused!
    </p>
    <div>
      <h3 class="text-xl">Procedures</h3>
      <ul class="list-decimal ml-4">
        <li>Noise Displayed</li>
        <li>
          A bounding box displayed on noise (so that you can focus on the
          position)
        </li>
        <li>Actual Image displayed</li>
        <li>Noise image will replace the actual image</li>
        <li>
          You will classify the object in the bounding box and select the class
        </li>
      </ul>
    </div>

    <p>
      <em><b>The estimated time for this HIT is about 40 seconds.</b></em>
    </p>

    <p>
      <em
        ><b class="text-red-500"
          >If you submit the HIT without completing the task, it will
          automatically be rejected. Please DO NOT press ENTER at any time
          during the HIT as this may automaticcaly submit the HIT without
          completing it.</b
        ></em
      >
    </p>
    <p class="text-red-500">
      We will monitor your accuracy, low accuracy can cause the job to be
      rejected.
    </p>

    <h3 class="text-2xl">Task Example</h3>
    <h3 align="center">This is an example of a</h3>

    <h2 align="center" style="color: red"><b>car</b></h2>

    <div align="center" class="bbox-container">
      <img
        id="imgExample"
        src="https://pilotexp.s3.ca-central-1.amazonaws.com/Examples_Experiment/ILSVRC2012_val_00014541.JPEG"
        style="height: 400px"
      />
    </div>

    <h3 class="text-xl font-bold">
      If there are multiple object in an image, there will be only one bounding
      box
    </h3>
    <img
      style="width: 30em"
      src="https://mturk-host.s3.us-east-2.amazonaws.com/instructions/classification/2-birds.png"
      alt=""
    />
    <h2 class="text-2xl">Live Demo</h2>
    <div>
      <task-view :image-data="qualificationTestData.slice(0, 4)" model="demo" />
    </div>

    <h2 class="text-2xl">Qualification Test</h2>
    <p>
      If this is the your first HIT of this project, you should first pass a
      simple qualification test where you are asked to pick the most appropriate
      class of 20 toy images.
    </p>
    <p>
      To prevent you from doing the test for every HIT, a Skip Code will be
      given to you. Please take note of it and you can use it to skip the test
      later.
    </p>
    <p>
      In order to pass the qualification test, you need to correctly identify
      <em>all 20 </em>images.
    </p>
    <p>
      If you have already passed this test in a previous HIT,you have the option
      to insert the qualification password here in order to skip the test:
    </p>
    <el-input placeholder="Skip Code" v-model="codeInput" />
    <el-button @click="check" class="mt-2 float-right">Check</el-button>
    <p>Click on the most appropriate class for this object in the box.</p>

    <p class="text-red-500">
      By clicking <strong>Proceed</strong>, the qualification test will start
      immediately, so please be ready.
    </p>

    <el-button
      class="mb-5 float-right"
      size="large"
      type="primary"
      @click="proceed"
      >Proceed</el-button
    >
  </main>
</template>
