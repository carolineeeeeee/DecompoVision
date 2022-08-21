<template>
  <h2 class="text-2xl">You Finished the Hit</h2>
  <h2 class="text-2xl">Thank You</h2>
</template>
<script lang="ts" setup>
import { onMounted } from "vue";
onMounted(() => {
  const customCrowdForm = document.getElementById("custom-crowd-form");
  customCrowdForm?.classList.remove("hidden");
});
</script>
