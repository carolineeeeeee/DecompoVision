<script setup lang="ts">
import { onMounted } from "vue";
import { RouterView, useRoute, useRouter } from "vue-router";
import { initInputElements } from "@/util/util";
import { taskImageData } from "@/util/constant";

const router = useRouter();

const route = useRoute();
// Enable this in production
onMounted(() => {
  console.log(route.fullPath);
  console.log(route);
  setTimeout(() => {
    if (import.meta.env.PROD) {
      router.push("/");
    }
  }, 1000)
  setTimeout(() => {
    initInputElements(taskImageData.map((data) => data.url));
  }, 2000);

});
</script>

<template>
  <div class="container mx-auto">
    <!-- <h1>App.vue</h1> -->
    <header>
      <h1 class="text-3xl font-bold">Mturk Classification Job</h1>
    </header>
    <!-- <nav>
      <RouterLink class="mr-2" to="/">
        <el-button>Home</el-button>
      </RouterLink>
      <RouterLink class="mr-2" to="/test">
        <el-button>Test</el-button>
      </RouterLink>
      <RouterLink class="mr-2" to="/task">
        <el-button>Task</el-button>
      </RouterLink>
      <RouterLink class="mr-2" to="/finish">
        <el-button>Finish</el-button>
      </RouterLink>
      <RouterLink class="mr-2" to="/sanity-check-fail">
        <el-button>Sanity Fail</el-button>
      </RouterLink>
    </nav> -->
    <RouterView />
  </div>
</template>
