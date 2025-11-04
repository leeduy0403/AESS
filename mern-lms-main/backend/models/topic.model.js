import mongoose from "mongoose";

const topicSchema = new mongoose.Schema(
  {
    content: {
      type: String,
      required: true,
    },
    questions: {
      type: Array,
      default: [],
    },
    userId: {
      type: String,
    },
    forumId: {
      type: String,
    },
  },
  { timestamps: true }
);

const Topic = mongoose.model("Topic", topicSchema);

export default Topic;
