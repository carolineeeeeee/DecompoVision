<script setup lang="ts">
import { RouterLink, RouterView } from "vue-router";
import { initInputElements } from "@/util/util";
import { onMounted } from "vue";
import { imageUrls } from "@/util/constant";
import { useRouter } from "vue-router";
const isDev = import.meta.env.DEV;
const router = useRouter();

onMounted(() => {
  setTimeout(() => {
    router.push("/");
    initInputElements(imageUrls);
  }, 2000);
});
</script>

<template>
  <div class="container mx-auto">
    <header>
      <h1 class="text-5xl font-extrabold">Mturk Bounding Box Drawing</h1>
      <nav class="mt-5 mb-3">
        <div v-if="isDev">
          <nav class="mt-5 mb-3">
            <RouterLink class="mr-3" to="/"
              ><el-button>Home</el-button></RouterLink
            >
            <RouterLink class="mr-3" to="/test"
              ><el-button>Qualification Test</el-button></RouterLink
            >
            <RouterLink class="mr-3" to="/task"
              ><el-button>Task</el-button></RouterLink
            >
            <RouterLink class="mr-3" to="/finish"
              ><el-button>Finish</el-button></RouterLink
            >
          </nav>
        </div>
      </nav>
    </header>

    <RouterView />
  </div>
</template>
