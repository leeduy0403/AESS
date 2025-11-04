import mongoose from "mongoose";

const forumSchema = new mongoose.Schema(
  {
    title: {
      type: String,
    },
    topics: {
      type: Array,
      default: [],
    },
    assignmentId: {
      type: String,
    },
    classId: {
      type: String,
    },
  },
  { timestamps: true }
);

const Forum = mongoose.model("Forum", forumSchema);

export default Forum;
