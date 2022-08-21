<script setup lang="ts">
import { ref } from "vue";
import { qualificationTestData, qualificationCode } from "@/util/constant";
import { useRouter } from "vue-router";

const router = useRouter();
const codeInput = ref();

const check = () => {
  if (qualificationCode === codeInput.value) {
    router.push("/task");
  } else {
    alert("Wrong Skip Code");
  }
};

const proceed = () => {
  router.push("/test");
};
</script>

<template>
  <main>
    <div class="instruction container">
      <h1 class="text-4xl font-bold">Instructions</h1>
      <p>
        In this job, you are asked to draw bounding boxes around each foreground
        object you see.
      </p>
      <p>
        There are 2 timeouts for each image, one image timeout and one bounding
        box timeout.
      </p>
      <p>
        Image Timeout is the total time you can spend on an image, e.g. 20
        seconds to select all foreground objects.
      </p>
      <p>
        Bounding Box Timeout is the time you can spend on a single bounding box,
        e.g. 3 seconds to detect and draw a bounding box
      </p>
      <p>
        Bounding Box Timeout is refreshed everytime you draw a new bounding box.
      </p>
      <p>
        When any of the 2 timeouts finishs, an image is finished and will switch
        to the next one.
      </p>
      <p>
        In other words, you must keep selecting objects until image timeout is
        done or there is no more objects (wait for Bounding Box Timeout to
        finish)
      </p>

      <div style="text-align: left">
        <h3 class="text-2xl font-semibold">Procedures</h3>
        <ul class="list-disc ml-4">
          <li>Image Displayed</li>
          <li>
            Keep drawing tight bounding boxes around all foreground objects you
            detect
          </li>
          <li>
            When there is no more foreground objects to select, wait for
            bounding box timeout
          </li>
        </ul>
      </div>

      <p>
        <em><b>The estimated time for this HIT is about 2 minutes.</b></em>
      </p>

      <p>
        <em
          ><b style="color: red"
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

      <h2 class="text-2xl font-semibold">Demo Video</h2>
      <video width="800" controls controlsList="nodownload">
        <source
          src="https://mturk-host.s3.us-east-2.amazonaws.com/instructions/detection/detection-video-demo.mp4"
          type="video/mp4"
        />
        Your browser does not support the video tag.
      </video>
      <h3 class="text-2xl font-semibold">Draw a tight box</h3>
      <img
        class="w-[60em]"
        src="https://mturk-host.s3.us-east-2.amazonaws.com/instructions/detection/tight-box.png"
        alt=""
      />
      <h3 class="text-2xl font-semibold">
        For an object with many objects on it (books on shelf), you just need to
        select the main object
      </h3>
      <img
        class="w-[20em]"
        src="https://mturk-host.s3.us-east-2.amazonaws.com/instructions/detection/book-shelf.png"
        alt=""
      />
      <h3 class="text-2xl font-semibold">
        If there are multiple objects, select all of them.
      </h3>
      <h3 class="text-2xl font-semibold">
        Partial Object should also be selected.
      </h3>
      <img
        class="w-[40em]"
        src="https://mturk-host.s3.us-east-2.amazonaws.com/instructions/detection/multi-horses.png"
        alt=""
      />

      <h2 class="text-2xl">Qualification Test</h2>
      <p>
        If this is the your first HIT of this project, you should first pass a
        simple qualification test where you are asked to pick the most
        appropriate class of 20 toy images.
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
        If you have already passed this test in a previous HIT,you have the
        option to insert the qualification password here in order to skip the
        test:
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
    </div>
  </main>
</template>
