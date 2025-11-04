import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { useEffect, useState } from "react";
import { AiOutlineUsergroupAdd } from "react-icons/ai";
import { Input } from "@/components/ui/input";
import { apiClient } from "@/lib/api-client";
import {
  CREATE_CHANNEL_ROUTE,
  GET_ALL_CONTACTS_ROUTE,
} from "@/utils/constants";
import { Button } from "@/components/ui/button";
import MultipleSelector from "@/components/ui/multiple-selector";
import { useSelector, useDispatch } from "react-redux";
import {
  setSelectedChatType,
  setSelectedChatData,
  addChannel,
} from "@/redux/chat/chatSlice";

function CreateChannel({ displayText, className }) {
  const dispatch = useDispatch();
  const [newChannelModal, setNewChannelModal] = useState(false);
  const [allContacts, setAllContacts] = useState([]);
  const [selectedContacts, setSelectedContacts] = useState([]);
  const [channelName, setChannelName] = useState("");

  useEffect(() => {
    const getData = async () => {
      const response = await apiClient.get(GET_ALL_CONTACTS_ROUTE, {
        withCredentials: true,
      });
      if (response.data.contacts) {
        setAllContacts(response.data.contacts);
      }
    };

    getData();
  }, []);

  const createChannel = async () => {
    try {
      //   console.log(selectedContacts.map((contact) => contact.value)); //? debug
      if (channelName.length > 0 && selectedContacts.length > 0) {
        const response = await apiClient.post(
          CREATE_CHANNEL_ROUTE,
          {
            name: channelName,
            members: selectedContacts.map((contact) => contact.value),
          },
          { withCredentials: true }
        );
        if (response.status === 201) {
          setChannelName("");
          setSelectedContacts([]);
          setNewChannelModal(false);
          dispatch(addChannel(response.data.channel));
        }
      }
    } catch (error) {
      console.log({ error });
    }
  };

  return (
    <>
      <Button
        className={
          className ||
          "group cursor-pointer bg-[#F8F8D5] border-2 border-black ml-2 transition-all duration-300 hover:bg-[#cfcfa1]"
        }
        onClick={() => setNewChannelModal(true)}
      >
        {displayText ? (
          <span className="text-center text-lg text-white">{displayText}</span>
        ) : (
          <AiOutlineUsergroupAdd className="cursor-pointer text-start font-light text-black transition-all duration-300" />
        )}
      </Button>
      <Dialog open={newChannelModal} onOpenChange={setNewChannelModal}>
        <DialogContent className="flex h-[400px] w-[500px] flex-col border-none bg-[#ffffff]">
          <DialogHeader>
            <DialogTitle className="text-center text-xl">
              Create new Group
            </DialogTitle>
            <DialogDescription></DialogDescription>
          </DialogHeader>
          <div>
            <Input
              placeholder="Group Name"
              className="text-medium rounded-lg border-2 border-black-20 bg-[#ebebeb] p-6 transition-all duration-100"
              onChange={(e) => setChannelName(e.target.value)}
              value={channelName}
            />
          </div>
          <div>
            <MultipleSelector
              className="rounded-lg border-none bg-[#ebebeb] py-2 pl-3 transition-all duration-300 focus:bg-white"
              defaultOptions={allContacts}
              placeholder="Search Contacts"
              value={selectedContacts}
              onChange={setSelectedContacts}
              emptyIndicator={
                <p className="text-center text-lg leading-10 text-black bg-white">
                  No results found.
                </p>
              }
            />
          </div>
          <div>
            <Button
              className="w-full text-white bg-[#26597C] hover:bg-[#1d4762] transition-all duration-300"
              onClick={createChannel}
            >
              Create Group
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}

export default CreateChannel;
