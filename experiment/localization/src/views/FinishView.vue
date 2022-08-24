<template>
  <h1 class="text-4xl font-bold">Finish</h1>
  <!-- <button @click="submit">Submit</button>
  <el-button @click="submit">Submit</el-button>
  <h2 class="text-2xl font-bold">Finish 2</h2> -->
</template>
<script setup lang="ts">
import { onMounted } from "vue";
import { nSuccess } from "@/util/notification";
import useResultStore from "@/stores/result";

const store = useResultStore();

onMounted(() => {
  nSuccess("Finished", "Congratulations, you've finished everything");
  const customCrowdForm = document.getElementById("custom-crowd-form");
  customCrowdForm?.classList.remove("hidden");
});

// const submit = () => {
//   const urlParams = new URLSearchParams(window.location.search);

//   // create the form element and point it to the correct endpoint
//   const form = document.createElement("form");
//   form.action = new URL(
//     "https://workersandbox.mturk.com/mturk/externalSubmit",
//     urlParams.get("turkSubmitTo") as string
//   ).href;
//   form.method = "post";

//   // attach the assignmentId
//   const inputAssignmentId = document.createElement("input");
//   inputAssignmentId.name = "assignmentId";
//   // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
//   inputAssignmentId.value = urlParams.get("assignmentId")!;
//   inputAssignmentId.hidden = true;
//   form.appendChild(inputAssignmentId);

//   // attach data I want to send back
//   let count = 0;
//   for (const [url, payload] of Object.entries(store.bboxes)) {
//     // const {url, scale, bboxes} = payload;
//     const resultInputEle = document.createElement("input");
//     resultInputEle.name = `result-${count}`;
//     resultInputEle.value = JSON.stringify(payload);
//     resultInputEle.hidden = true;
//     form.appendChild(resultInputEle);
//     count++;
//   }
//   console.log(form);

//   // const inputCoordinates = document.createElement("input");
//   // inputCoordinates.name = "coordinates";
//   // inputCoordinates.value = JSON.stringify(coordinates);
//   // inputCoordinates.hidden = true;
//   // form.appendChild(inputCoordinates);

//   // attach the form to the HTML document and trigger submission
//   document.body.appendChild(form);
//   form.submit();
//   // console.log("submit");
//   // console.log(document.querySelector("#custom-crowd-form"));
//   // (document.querySelector("#custom-crowd-form") as HTMLFormElement).submit();
// };
</script>
