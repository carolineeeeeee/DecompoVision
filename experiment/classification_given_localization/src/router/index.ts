import { createRouter, createWebHistory } from "vue-router";
import HomeView from "@/views/HomeView.vue";
import { taskImageData } from "@/util/constant";

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: "/",
      name: "home",
      component: HomeView,
    },
    {
      path: "/test",
      name: "test",
      component: () => import("@/views/TestView.vue"),
    },
    {
      path: "/sanity-check-fail",
      name: "sanity-check-fail",
      component: () => import("@/views/SanityFail.vue"),
    },
    {
      path: "/task",
      name: "task",
      // route level code-splitting
      // this generates a separate chunk (About.[hash].js) for this route
      // which is lazy-loaded when the route is visited.
      component: () => import("@/views/TaskView.vue"),
      props: {
        imageData: taskImageData,
        mode: "task",
        canvasContainerId: "task-canvas-container",
      },
    },
    {
      path: "/finish",
      name: "finish",
      // route level code-splitting
      // this generates a separate chunk (About.[hash].js) for this route
      // which is lazy-loaded when the route is visited.
      component: () => import("@/views/FinishView.vue"),
    },
  ],
});

export default router;
