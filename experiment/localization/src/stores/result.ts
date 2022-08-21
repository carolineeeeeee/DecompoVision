import type { BBox } from "@/util/types";
import { defineStore } from "pinia";

export type Payload = {
  url: string;
  scale: number;
  bboxes: BBox[];
};

/**
 * key is url
 */
export interface StateType {
  bboxes: {
    [key: string]: Payload;
  };
}

const useResultStore = defineStore({
  id: "result",
  state: (): StateType => ({
    bboxes: {},
  }),
  getters: {},
  actions: {
    addImageBBox(payload: Payload) {
      console.log("addImageBBox");
      console.log(payload);
      this.bboxes[payload.url] = payload;
    },
    clear() {
      this.bboxes = {};
    },
  },
});

export default useResultStore;
