import { useSelector } from "react-redux";
import NewDM from "../contacts-container/components/new-dm";
import CreateChannel from "../contacts-container/components/create-channel";

function EmptyChatContainer() {
  const { currentUser } = useSelector((state) => state.user);

  return (
    <div className="hidden flex-1 flex-col items-center justify-center transition-all duration-1000 md:flex border-2 border-black bg-white rounded-md">
      <div className="mt-10 flex flex-col items-center gap-5 text-center text-3xl text-black text-opacity-80 transition-all duration-300 lg:text-4xl">
        <div className="mx-5 flex flex-wrap gap-2 items-center justify-center bg-clip-content">
          <span className="font-semibold">Welcome</span>
          <span className="font-semibold text-clip">{currentUser?.name}</span>
        </div>
        <p className="text-2xl font-semibold">Select existing contact or </p>
        <div className="flex flex-col gap-3 items-center justify-center xl:flex-row">
          <NewDM
            className="bg-[#26597C] hover:bg-[#1d4762] p-5"
            displayText="Select a new Contact"
          />
          <p className="text-xl"> or </p>
          <CreateChannel
            className="bg-[#26597C] hover:bg-[#1d4762] p-5"
            displayText="Create new Group"
          />
        </div>
      </div>
    </div>
  );
}

export default EmptyChatContainer;
