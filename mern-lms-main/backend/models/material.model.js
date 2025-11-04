import mongoose from "mongoose";

const materialSchema = new mongoose.Schema(
  {
    description: {
      type: String,
    },
    isHidden: {
      type: Boolean,
      default: false,
    },
    sectionId: {
      type: String,
    },
    assignmentId: {
      type: String,
    },
    uploadBy: {
      type: String,
      required: true,
    },
    materialUrls: {
      type: Array,
      required: true,
      default: [],
    },
    nameFiles: {
      type: Array,
      required: true,
      default: [],
    },
  },
  { timestamps: true }
);

const Material = mongoose.model("Material", materialSchema);

export default Material;
