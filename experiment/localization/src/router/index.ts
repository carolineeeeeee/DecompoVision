import { createRouter, createWebHistory } from "vue-router";
import HomeView from "../views/HomeView.vue";
import {
  imageUrls,
  countdownTime,
  bboxTimeouts,
  totalWaitingTimePerImage,
  bboxImageTimeouts,
} from "@/util/constant";

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: "/",
      name: "Home",
      component: HomeView,
    },
    {
      path: "/test",
      name: "Test",
      component: () => import("../views/TestView.vue"),
    },
    {
      path: "/sanity-check-fail",
      name: "sanity-check-fail",
      component: () => import("@/views/SanityFail.vue"),
    },
    {
      path: "/task",
      name: "Task",
      // route level code-splitting
      // this generates a separate chunk (About.[hash].js) for this route
      // which is lazy-loaded when the route is visited.
      component: () => import("../views/TaskView.vue"),
      props: {
        imageUrls,
        countdownTime,
        bboxTimeouts: imageUrls.map(() => countdownTime),
        bboxImageTimeouts: imageUrls.map(() => totalWaitingTimePerImage),
        mode: "task",
        canvasContainerId: "task-canvas-container",
      },
    },
    {
      path: "/finish",
      name: "Finish",
      // route level code-splitting
      // this generates a separate chunk (About.[hash].js) for this route
      // which is lazy-loaded when the route is visited.
      component: () => import("../views/FinishView.vue"),
    },
  ],
});

export default router;
