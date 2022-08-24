/**
 * This store is for storing classification result
 */
import { defineStore } from "pinia";

interface StateType {
  classes: string[];
  imageUrls: string[];
}

const useResultStore = defineStore({
  id: "result",
  state: (): StateType => ({
    classes: [],
    imageUrls: [],
  }),
  actions: {
    appendClass(_class: string) {
      this.classes.push(_class);
    },
    appendImageUrl(_url: string) {
      this.imageUrls.push(_url);
    },
    clear() {
      this.classes = [];
      this.imageUrls = [];
    },
  },
});

export default useResultStore;
