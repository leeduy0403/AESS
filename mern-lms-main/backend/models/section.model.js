import mongoose from "mongoose";

const sectionSchema = new mongoose.Schema(
  {
    name: {
      type: String,
      required: true,
    },
    description: {
      type: String,
    },
    isHidden: {
      type: Boolean,
      default: false,
    },
    materials: {
      type: Array,
      default: [],
    },
  },
  { timestamps: true }
);

const Section = mongoose.model("Section", sectionSchema);

export default Section;
