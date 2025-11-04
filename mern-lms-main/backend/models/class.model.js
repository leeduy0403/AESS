import mongoose from "mongoose";

const classSchema = new mongoose.Schema(
  {
    name: {
      type: String,
      required: true,
    },
    courseId: {
      type: String,
      required: true,
    },
    educators: {
      type: Array,
      default: [],
    },
    students: {
      type: Array,
      default: [],
    },
    sections: {
      type: Array,
      default: [],
    },
    assignments: {
      type: Array,
      default: [],
    },
    forums: {
      type: Array,
      default: [],
    },
    subjectName: {
      type: String,
    },
    subjectCode: {
      type: String,
    },
    semester: {
      type: String,
    },
  },
  { timestamps: true }
);

const Class = mongoose.model("Class", classSchema);

export default Class;
